import os, uuid, asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from youtube_utils import (
    get_transcript_chunks, get_video_info, get_live_transcript_since, extract_video_id,
    extract_frame_at, extract_clip_frames,
    download_video_local, extract_frame_local, extract_frames_local_batch,
    transcribe_audio, extract_audio_from_video, detect_whistle_timestamps,
)
from claude_analyzer import (
    analyze_chunk, analyze_frame_for_stats, analyze_play_sequence,
    quick_score_check, analyze_transcription_chunks,
    analyze_referee_foul_sequence, analyze_timeout_screen,
    extract_jersey_numbers_ocr, scan_frame_for_jerseys,
    detect_team_jersey_colors, detect_substitution,
)
from tracker import (
    TeamClassifier, PlayerTracker, BallTracker, EventDetector,
    analyze_clip as cv_analyze_clip,
    analyze_clip_yolo,
    scan_warmup_for_jerseys as cv_scan_warmup,
    extract_player_crop,
    YOLO_AVAILABLE,
)

app = FastAPI(title="Titans Basketball Auto Tracker")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")


class WSManager:
    def __init__(self):
        self.connections: dict[str, WebSocket] = {}

    async def connect(self, sid: str, ws: WebSocket):
        await ws.accept()
        self.connections[sid] = ws

    def disconnect(self, sid: str):
        self.connections.pop(sid, None)

    async def send(self, sid: str, data: dict):
        ws = self.connections.get(sid)
        if ws:
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect(sid)


manager = WSManager()
active_tasks: dict[str, asyncio.Task] = {}
live_sessions: dict[str, dict] = {}  # session_id -> {url, last_ts, players}


class StartReq(BaseModel):
    url: str
    session_id: str
    players: list[str]

class StopReq(BaseModel):
    session_id: str

class VisionReq(BaseModel):
    url: str
    session_id: str
    players: list[str]
    timestamp: int = 60        # video timestamp in seconds to analyze
    scan_interval: int = 0     # 0 = single frame, >0 = auto-scan every N seconds

class VisionScanReq(BaseModel):
    url: str
    session_id: str
    players: list[str]
    interval: int = 45         # seconds between frame captures

class SmartScanReq(BaseModel):
    url: str
    session_id: str
    players: list[str]
    jersey_map: dict = {}      # {"7": "Joseph Gabay", "11": "Aaron Breziner"}
    poll_interval: int = 12    # seconds between score checks
    start_ts: int = 60         # skip intro

class FullAutoReq(BaseModel):
    url: str
    session_id: str
    players: list[str]
    jersey_map: dict = {}
    score_interval: int = 3     # seconds between local score checks (3 = tight window)
    player_profiles: dict = {}  # {"Joseph Gabay": "short guard, shoots 3PT from right wing"}
    titans_jersey_color: str = "gray/white"
    rival_jersey_color: str = "colored"


@app.get("/")
async def root():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())


@app.post("/api/start")
async def start_tracking(req: StartReq, bg: BackgroundTasks):
    if not os.getenv("ANTHROPIC_API_KEY"):
        return {"error": "ANTHROPIC_API_KEY missing in .env"}

    if req.session_id in active_tasks:
        active_tasks[req.session_id].cancel()
        await asyncio.sleep(0.1)

    task = asyncio.create_task(
        pipeline(req.url, req.session_id, req.players)
    )
    active_tasks[req.session_id] = task
    return {"status": "started"}


@app.post("/api/stop")
async def stop_tracking(req: StopReq):
    task = active_tasks.pop(req.session_id, None)
    if task:
        task.cancel()
    live_sessions.pop(req.session_id, None)
    return {"status": "stopped"}


@app.websocket("/ws/{session_id}")
async def ws_endpoint(ws: WebSocket, session_id: str):
    await manager.connect(session_id, ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(session_id)


async def pipeline(url: str, sid: str, players: list[str]):
    try:
        await manager.send(sid, {"type": "status", "msg": "Fetching video info...", "level": "info"})

        info = await get_video_info(url)
        is_live = False
        if info:
            is_live = info.get("is_live", False)
            await manager.send(sid, {
                "type": "video_info",
                "title": info.get("title", ""),
                "duration": info.get("duration", 0),
                "is_live": is_live,
            })

        if is_live:
            await run_live_pipeline(url, sid, players)
        else:
            await run_vod_pipeline(url, sid, players)

    except asyncio.CancelledError:
        await manager.send(sid, {"type": "status", "msg": "Tracking stopped.", "level": "warn"})
    except Exception as e:
        await manager.send(sid, {"type": "error", "msg": f"Error: {str(e)}"})
    finally:
        active_tasks.pop(sid, None)


async def run_vod_pipeline(url: str, sid: str, players: list[str]):
    await manager.send(sid, {"type": "status", "msg": "Fetching transcript...", "level": "info"})
    chunks = await get_transcript_chunks(url)

    if not chunks:
        await manager.send(sid, {
            "type": "status",
            "msg": "⚠ No transcript found. Use manual tracking or try a different video.",
            "level": "warn"
        })
        return

    total = len(chunks)
    await manager.send(sid, {
        "type": "status",
        "msg": f"Analyzing {total} segments with AI... This may take 1-2 minutes.",
        "level": "info"
    })

    for i, chunk in enumerate(chunks):
        events = await analyze_chunk(chunk["text"], chunk["start_fmt"], players)

        for ev in events:
            await manager.send(sid, {
                "type": "ai_event",
                "id": str(uuid.uuid4())[:8],
                "video_ts": chunk["start_fmt"],
                **ev
            })

        pct = int(((i + 1) / total) * 100)
        await manager.send(sid, {"type": "progress", "pct": pct, "chunk": i + 1, "total": total})
        await asyncio.sleep(0.15)

    await manager.send(sid, {"type": "status", "msg": "✓ AI analysis complete!", "level": "success"})
    await manager.send(sid, {"type": "progress", "pct": 100, "done": True})


async def run_live_pipeline(url: str, sid: str, players: list[str]):
    last_ts = 0.0
    live_sessions[sid] = {"url": url, "players": players, "last_ts": last_ts}

    await manager.send(sid, {
        "type": "status",
        "msg": "Live stream detected. Polling for new captions every 30s...",
        "level": "info"
    })

    while sid in live_sessions:
        await asyncio.sleep(30)
        if sid not in live_sessions:
            break

        new_segs = await get_live_transcript_since(url, last_ts)
        if new_segs:
            text = " ".join(s.get("text", "") for s in new_segs)
            if new_segs:
                last_ts = new_segs[-1].get("start", last_ts) + new_segs[-1].get("duration", 0)
                live_sessions[sid]["last_ts"] = last_ts

            events = await analyze_chunk(text, "LIVE", players)
            for ev in events:
                await manager.send(sid, {
                    "type": "ai_event",
                    "id": str(uuid.uuid4())[:8],
                    "video_ts": "LIVE",
                    **ev
                })

        await manager.send(sid, {"type": "heartbeat", "last_ts": last_ts})


# ── Vision Analysis Endpoints ──────────────────────────────────────────────

@app.post("/api/analyze-frame")
async def analyze_single_frame(req: VisionReq, bg: BackgroundTasks):
    """Analyze one frame at a specific timestamp via Claude Vision."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return {"error": "ANTHROPIC_API_KEY missing"}
    bg.add_task(vision_single_frame, req.url, req.session_id, req.players, req.timestamp)
    return {"status": "started", "timestamp": req.timestamp}


@app.post("/api/start-vision-scan")
async def start_vision_scan(req: VisionScanReq, bg: BackgroundTasks):
    """Start automated frame scanning every N seconds."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return {"error": "ANTHROPIC_API_KEY missing"}

    key = f"vision_{req.session_id}"
    if key in active_tasks:
        active_tasks[key].cancel()

    task = asyncio.create_task(
        vision_scan_pipeline(req.url, req.session_id, req.players, req.interval)
    )
    active_tasks[key] = task
    return {"status": "scanning", "interval": req.interval}


@app.post("/api/stop-vision-scan")
async def stop_vision_scan(req: StopReq):
    key = f"vision_{req.session_id}"
    task = active_tasks.pop(key, None)
    if task:
        task.cancel()
    return {"status": "stopped"}


async def vision_single_frame(url: str, sid: str, players: list[str], timestamp: int):
    frames_dir = f"/tmp/frames_{sid}"
    ts_fmt = f"{timestamp//60}:{timestamp%60:02d}"

    await manager.send(sid, {
        "type": "status",
        "msg": f"📷 Downloading frame at {ts_fmt}... (~10s)",
        "level": "info"
    })

    frame_path = await extract_frame_at(url, timestamp, frames_dir)
    if not frame_path:
        await manager.send(sid, {"type": "status", "msg": f"⚠ Could not extract frame at {ts_fmt}", "level": "warn"})
        return

    await manager.send(sid, {"type": "status", "msg": f"🔍 Analyzing frame with Claude Vision...", "level": "info"})
    result = await analyze_frame_for_stats(frame_path, players, f"Video time {ts_fmt}")

    try:
        os.remove(frame_path)
        os.rmdir(frames_dir)
    except Exception:
        pass

    # Send score update
    if result.get("titans_score") is not None or result.get("rival_score") is not None:
        await manager.send(sid, {
            "type": "score_update",
            "titans_score": result.get("titans_score"),
            "rival_score": result.get("rival_score"),
            "quarter": result.get("quarter"),
            "clock": result.get("clock"),
            "video_ts": ts_fmt,
        })

    # Send any player events detected in graphics
    for ev in result.get("player_events", []):
        await manager.send(sid, {
            "type": "ai_event",
            "id": str(uuid.uuid4())[:8],
            "source": "vision",
            "video_ts": ts_fmt,
            **ev
        })

    text_on_screen = result.get("text_on_screen", "")
    msg = f"📷 Frame @{ts_fmt}: "
    if result.get("titans_score") is not None:
        msg += f"Score {result.get('titans_score')}-{result.get('rival_score')} {result.get('quarter','')} {result.get('clock','')} | "
    msg += text_on_screen[:60] if text_on_screen else "(no text detected)"
    await manager.send(sid, {"type": "status", "msg": msg, "level": "success"})


async def vision_scan_pipeline(url: str, sid: str, players: list[str], interval: int):
    """Auto-scan video frames at regular intervals."""
    info = await get_video_info(url)
    duration = info.get("duration", 0) if info else 0

    if not duration:
        await manager.send(sid, {"type": "status", "msg": "⚠ Could not get video duration for scan", "level": "warn"})
        return

    frames_dir = f"/tmp/frames_{sid}_scan"
    start_ts = 30  # Skip first 30s (usually pre-game)
    timestamps = list(range(start_ts, min(duration, 7200), interval))
    total = len(timestamps)

    await manager.send(sid, {
        "type": "status",
        "msg": f"📷 Vision scan started — {total} frames to analyze ({interval}s interval)",
        "level": "info"
    })

    try:
        for i, ts in enumerate(timestamps):
            if f"vision_{sid}" not in active_tasks:
                break

            frame_path = await extract_frame_at(url, ts, frames_dir)
            if frame_path:
                result = await analyze_frame_for_stats(frame_path, players, f"Video {ts//60}:{ts%60:02d}")
                try:
                    os.remove(frame_path)
                except Exception:
                    pass

                if result.get("titans_score") is not None or result.get("rival_score") is not None:
                    await manager.send(sid, {
                        "type": "score_update",
                        "titans_score": result.get("titans_score"),
                        "rival_score": result.get("rival_score"),
                        "quarter": result.get("quarter"),
                        "clock": result.get("clock"),
                        "video_ts": f"{ts//60}:{ts%60:02d}",
                    })

                for ev in result.get("player_events", []):
                    await manager.send(sid, {
                        "type": "ai_event",
                        "id": str(uuid.uuid4())[:8],
                        "source": "vision",
                        "video_ts": f"{ts//60}:{ts%60:02d}",
                        **ev
                    })

            pct = int(((i + 1) / total) * 100)
            await manager.send(sid, {"type": "vision_progress", "pct": pct, "frame": i + 1, "total": total})
            await asyncio.sleep(0.3)

        await manager.send(sid, {"type": "status", "msg": "✓ Vision scan complete!", "level": "success"})
    except asyncio.CancelledError:
        pass
    finally:
        try:
            os.rmdir(frames_dir)
        except Exception:
            pass
        active_tasks.pop(f"vision_{sid}", None)


# ── Smart Scan — fully automatic play-by-play via multi-frame analysis ─────

@app.post("/api/start-smart-scan")
async def start_smart_scan(req: SmartScanReq):
    if not os.getenv("ANTHROPIC_API_KEY"):
        return {"error": "ANTHROPIC_API_KEY missing"}
    key = f"smart_{req.session_id}"
    if key in active_tasks:
        active_tasks[key].cancel()
    task = asyncio.create_task(
        smart_scan_pipeline(req.url, req.session_id, req.players, req.jersey_map, req.poll_interval, req.start_ts)
    )
    active_tasks[key] = task
    return {"status": "started"}


@app.post("/api/stop-smart-scan")
async def stop_smart_scan(req: StopReq):
    task = active_tasks.pop(f"smart_{req.session_id}", None)
    if task:
        task.cancel()
    return {"status": "stopped"}


async def smart_scan_pipeline(url: str, sid: str, players: list[str], jersey_map: dict, poll_interval: int, start_ts: int):
    """
    Fully automatic pipeline:
    1. Poll a score-check frame every `poll_interval` seconds
    2. When score changes — extract 6-frame clip from that window
    3. Send all 6 frames to Claude Vision to identify who scored/fouled
    4. Emit confirmed events via WebSocket
    Also builds a jersey-number map from what Claude reads during the game.
    """
    info = await get_video_info(url)
    duration = int(info.get("duration", 0)) if info else 0
    if not duration:
        await manager.send(sid, {"type": "status", "msg": "⚠ Could not get video duration", "level": "warn"})
        return

    frames_dir = f"/tmp/smart_{sid}"
    os.makedirs(frames_dir, exist_ok=True)

    learned_jerseys = dict(jersey_map)   # grows as Claude spots numbers during the game
    last_score = {"titans": None, "rival": None}
    current_ts = start_ts
    total_secs = min(duration, 7200)

    await manager.send(sid, {
        "type": "status",
        "msg": f"🤖 Smart Scan started — polling every {poll_interval}s, full auto-detection active",
        "level": "info"
    })
    await manager.send(sid, {"type": "smart_scan_started"})

    try:
        while current_ts < total_secs and f"smart_{sid}" in active_tasks:
            ts_fmt = f"{current_ts//60}:{current_ts%60:02d}"

            # ── Step 1: quick score check (single frame) ──────────────────
            frame_path = await extract_frame_at(url, current_ts, frames_dir)
            if not frame_path:
                current_ts += poll_interval
                await asyncio.sleep(0.2)
                continue

            score_data = await analyze_frame_for_stats(frame_path, players, f"@{ts_fmt}")
            try:
                os.remove(frame_path)
            except Exception:
                pass

            cur_titans = score_data.get("titans_score")
            cur_rival  = score_data.get("rival_score")
            rival_name = score_data.get("rival_name") or "Rival"

            # Broadcast score update whenever we have valid data
            if cur_titans is not None or cur_rival is not None:
                await manager.send(sid, {
                    "type": "score_update",
                    "titans_score": cur_titans,
                    "rival_score": cur_rival,
                    "quarter": score_data.get("quarter"),
                    "clock": score_data.get("clock"),
                    "video_ts": ts_fmt,
                })

            # ── Step 2: detect score change ───────────────────────────────
            titans_delta = 0
            rival_delta  = 0
            if last_score["titans"] is not None and cur_titans is not None:
                titans_delta = cur_titans - last_score["titans"]
            if last_score["rival"] is not None and cur_rival is not None:
                rival_delta = cur_rival - last_score["rival"]

            score_changed = titans_delta > 0 or rival_delta > 0

            if last_score["titans"] is not None:
                await manager.send(sid, {
                    "type": "scan_tick",
                    "video_ts": ts_fmt,
                    "titans": cur_titans,
                    "rival": cur_rival,
                    "changed": score_changed,
                    "pct": int(current_ts / total_secs * 100),
                })

            # Update known score
            if cur_titans is not None:
                last_score["titans"] = cur_titans
            if cur_rival is not None:
                last_score["rival"] = cur_rival

            # ── Step 3: if score changed → analyze the play ───────────────
            if score_changed:
                change_ts = max(start_ts, current_ts - poll_interval + 2)
                change_fmt = f"{change_ts//60}:{change_ts%60:02d}"

                await manager.send(sid, {
                    "type": "status",
                    "msg": f"⚡ Score change @{change_fmt}! Titans+{titans_delta} Rival+{rival_delta} — analyzing play...",
                    "level": "info"
                })

                clip_frames = await extract_clip_frames(url, change_ts, frames_dir, num_frames=6)

                if clip_frames:
                    events, new_jerseys, play_desc = await analyze_play_sequence(
                        clip_frames, players, learned_jerseys,
                        {"titans": last_score["titans"] - titans_delta if last_score["titans"] is not None else 0,
                         "rival":  last_score["rival"]  - rival_delta  if last_score["rival"]  is not None else 0},
                        {"titans": last_score["titans"] or 0, "rival": last_score["rival"] or 0},
                        change_fmt,
                    )

                    # Update learned jersey map
                    if new_jerseys:
                        learned_jerseys.update(new_jerseys)
                        await manager.send(sid, {"type": "jersey_update", "map": learned_jerseys})

                    for ev in events:
                        player = ev.get("player", "UNKNOWN")
                        # Resolve UNKNOWN_TITANS to best guess if only one team scored
                        if player == "UNKNOWN_TITANS" and titans_delta > 0:
                            player = "Titans (unknown)"
                        conf = ev.get("confidence", 0)
                        await manager.send(sid, {
                            "type": "ai_event",
                            "id": str(uuid.uuid4())[:8],
                            "source": "smart_vision",
                            "video_ts": change_fmt,
                            "player": player,
                            "team": ev.get("team", "titans"),
                            "stat": ev.get("stat", ""),
                            "confidence": conf,
                            "quote": ev.get("reasoning", play_desc),
                        })

                    # Clean up clip frames
                    for f in clip_frames:
                        try:
                            os.remove(f)
                        except Exception:
                            pass

                    if play_desc:
                        await manager.send(sid, {
                            "type": "status",
                            "msg": f"🎬 {change_fmt}: {play_desc}",
                            "level": "success"
                        })

            current_ts += poll_interval
            await asyncio.sleep(0.3)

        await manager.send(sid, {"type": "status", "msg": "✅ Smart Scan complete! All plays detected.", "level": "success"})
        await manager.send(sid, {"type": "smart_scan_done", "learned_jerseys": learned_jerseys})

    except asyncio.CancelledError:
        pass
    finally:
        try:
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)
        except Exception:
            pass
        active_tasks.pop(f"smart_{sid}", None)


# ── Full Auto Pipeline ────────────────────────────────────────────────────

@app.post("/api/full-auto")
async def full_auto(req: FullAutoReq):
    if not os.getenv("ANTHROPIC_API_KEY"):
        return {"error": "ANTHROPIC_API_KEY missing"}
    key = f"auto_{req.session_id}"
    if key in active_tasks:
        active_tasks[key].cancel()
    task = asyncio.create_task(
        full_auto_pipeline(
            req.url, req.session_id, req.players, req.jersey_map,
            req.score_interval, req.player_profiles,
            req.titans_jersey_color, req.rival_jersey_color,
        )
    )
    active_tasks[key] = task
    return {"status": "started"}


@app.post("/api/stop-full-auto")
async def stop_full_auto(req: StopReq):
    task = active_tasks.pop(f"auto_{req.session_id}", None)
    if task:
        task.cancel()
    return {"status": "stopped"}


async def _send(sid, **kwargs):
    await manager.send(sid, kwargs)


async def full_auto_pipeline(
    url: str,
    sid: str,
    players: list[str],
    jersey_map: dict,
    score_interval: int = 3,
    player_profiles: dict | None = None,
    titans_jersey_color: str = "gray/white",
    rival_jersey_color: str = "colored",
):
    """
    5-phase Copa Talento pipeline — works 100% without audio commentary.
    Phase 0: Warmup scan (first 10 min, close-up shots → build jersey map)
    Phase 1: Jersey color detection from first readable frame
    Phase 2: Audio whistle detection (referee whistles from ambient gym sound)
    Phase 3: Scoreboard scan every 3s → play analysis with crop+zoom + color ID
    Phase 4: Quarter-break stat screen scanning (between Q1/Q2/Q3/Q4)
    """
    import shutil

    work_dir = f"/tmp/auto_{sid}"
    os.makedirs(work_dir, exist_ok=True)
    video_path  = os.path.join(work_dir, "game.mp4")
    audio_path  = os.path.join(work_dir, "game_audio.mp3")
    learned_jerseys      = dict(jersey_map)
    player_profiles      = player_profiles or {}
    last_score           = {"titans": None, "rival": None}
    last_quarter         = None
    last_clock           = None
    clock_same_count     = 0
    timeout_frames_analyzed = set()
    analyzed_whistles    = set()
    titans_color         = titans_jersey_color
    rival_color          = rival_jersey_color
    quarter_break_scanned = set()
    # Minutes tracking
    on_court_log         = {}
    on_court_since       = {}
    total_minutes        = {}
    # CV Tracking: map YOLO/OpenCV track IDs → player names
    warmup_track_to_player = {}   # {track_id: player_name}
    cv_classifier          = TeamClassifier(titans_jersey_color, rival_jersey_color)

    async def status(msg, level="info"):
        await manager.send(sid, {"type": "status", "msg": msg, "level": level})

    async def emit_event(source, video_ts, player, team, stat, confidence, quote):
        if player == "UNKNOWN_TITANS":
            player = "Titans (sin identificar)"
        await manager.send(sid, {
            "type": "ai_event",
            "id": str(uuid.uuid4())[:8],
            "source": source,
            "video_ts": video_ts,
            "player": player,
            "team": team,
            "stat": stat,
            "confidence": confidence,
            "quote": quote,
        })

    try:
        # ─── PHASE 0: Download video ───────────────────────────────────────
        await status("⬇ Downloading game video (this takes a few minutes)...")
        await manager.send(sid, {"type": "auto_phase", "phase": "download"})

        download_ok = await download_video_local(url, video_path)
        if not download_ok:
            await status("❌ Could not download video. Check URL.", "error")
            return

        info = await get_video_info(url)
        duration = int(info.get("duration", 0)) if info else 0
        if not duration:
            await status("❌ Could not read video duration.", "error")
            return

        loop = asyncio.get_event_loop()
        size_mb = os.path.getsize(video_path) // 1048576
        await status(f"✓ Video downloaded ({size_mb}MB, {duration//60}m {duration%60}s)")

        # ─── PHASE 1: Detect jersey colors from first readable frame ──────
        first_frame = os.path.join(work_dir, "first_frame.jpg")
        ok = await loop.run_in_executor(None, extract_frame_local, video_path, 90, first_frame)
        if ok and not jersey_map:
            await status("🎨 Detecting team jersey colors...")
            colors = await detect_team_jersey_colors(first_frame)
            if colors.get("titans_color"):
                titans_color = colors["titans_color"]
                rival_color  = colors.get("rival_color", rival_color)
                rival_name   = colors.get("rival_name", "")
                await status(f"✓ Titans = {titans_color} jerseys | {rival_name or 'Rival'} = {rival_color}")

        # ─── PHASE 2: Warmup scan (first 10 min → jersey number extraction) ──
        warmup_end = min(600, duration // 2)
        if warmup_end > 60:
            await status(f"🔍 Warmup scan: scanning first {warmup_end//60}m for jersey numbers...")
            await manager.send(sid, {"type": "auto_phase", "phase": "warmup"})

            # Layer A: CV scan for close-up player crops (fast, no API call)
            if YOLO_AVAILABLE:
                await status("🤖 YOLO player tracking active — highest accuracy mode", "success")
            cv_close_ups = await loop.run_in_executor(
                None, cv_scan_warmup, video_path, warmup_end, cv_classifier, 15
            )
            if cv_close_ups:
                await status(f"📸 CV warmup: {len(cv_close_ups)} close-up player crops found for jersey reading")
                # Feed best close-ups to Claude for jersey number reading
                # Group by track_id, pick the largest bbox (most readable)
                by_track = {}
                for cu in cv_close_ups:
                    tid = cu["track_id"]
                    if tid not in by_track or cu["bbox"][3] > by_track[tid]["bbox"][3]:
                        by_track[tid] = cu
                # Ask Claude to read jersey numbers from the best crops
                for tid, cu in list(by_track.items())[:10]:  # max 10 API calls
                    crop = await loop.run_in_executor(
                        None, extract_player_crop, video_path, cu["frame_idx"], cu["bbox"], 6
                    )
                    if crop is not None:
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=work_dir) as tf:
                            import cv2 as _cv2
                            _cv2.imwrite(tf.name, crop)
                            crop_path = tf.name
                        warmup_result = await scan_frame_for_jerseys(crop_path, players, learned_jerseys)
                        try: os.remove(crop_path)
                        except: pass
                        updates = warmup_result.get("jersey_map_updates", {})
                        for jnum, pname in updates.items():
                            if jnum not in learned_jerseys:
                                learned_jerseys[jnum] = pname
                                warmup_track_to_player[tid] = pname
                                await status(f"✓ CV+Claude: #{jnum} = {pname}", "success")
                                await manager.send(sid, {"type": "jersey_update", "map": learned_jerseys})
                        await asyncio.sleep(0.1)

            # Layer B: Claude warmup scan (whole frames for remaining jerseys)
            warmup_timestamps = list(range(30, warmup_end, 30))
            warmup_frame = os.path.join(work_dir, "warmup_frame.jpg")
            jerseys_found_warmup = 0
            for wt in warmup_timestamps:
                if f"auto_{sid}" not in active_tasks:
                    break
                ok = await loop.run_in_executor(None, extract_frame_local, video_path, wt, warmup_frame)
                if ok:
                    result = await scan_frame_for_jerseys(warmup_frame, players, learned_jerseys)
                    updates = result.get("jersey_map_updates", {})
                    if updates:
                        new_found = {k: v for k, v in updates.items() if k not in learned_jerseys}
                        if new_found:
                            learned_jerseys.update(new_found)
                            jerseys_found_warmup += len(new_found)
                            for num, name in new_found.items():
                                await status(f"✓ Warmup: found #{num} = {name}", "success")
                            await manager.send(sid, {"type": "jersey_update", "map": learned_jerseys})
                await asyncio.sleep(0.2)
            await status(f"✓ Warmup scan done — {jerseys_found_warmup} new jersey numbers found")

        # ─── PHASE 3: Audio whistle detection ────────────────────────────
        await status("🎵 Extracting audio for whistle/foul detection...")
        audio_ok = await loop.run_in_executor(None, extract_audio_from_video, video_path, audio_path)

        whistle_events = []
        whisper_task = None
        if audio_ok:
            whistle_events = await loop.run_in_executor(None, detect_whistle_timestamps, audio_path)
            w_count = sum(1 for e in whistle_events if e["type"] == "whistle")
            c_count = sum(1 for e in whistle_events if e["type"] == "cheer")
            await status(f"🎵 Audio: {w_count} referee whistles + {c_count} crowd cheers indexed")
            await manager.send(sid, {"type": "whistle_events", "count": w_count, "cheer_count": c_count})
            whisper_task = asyncio.create_task(transcribe_audio(url, work_dir))
        else:
            await status("⚠ Could not extract audio — visual analysis only", "warn")

        # ─── PHASE 4: Scoreboard scan every score_interval seconds ───────
        await status(f"📡 Main scan: every {score_interval}s for {duration//60}m...")
        await manager.send(sid, {"type": "auto_phase", "phase": "scanning"})
        whistle_set = {int(e["ts"]): e["type"] for e in whistle_events}

        timestamps = list(range(60, min(duration, 7200), score_interval))
        total = len(timestamps)
        frame_path = os.path.join(work_dir, "current_frame.jpg")

        for i, ts in enumerate(timestamps):
            if f"auto_{sid}" not in active_tasks:
                break

            # ── Extract frame (fast local ffmpeg, ~0.04s) ─────────────────
            ok = await loop.run_in_executor(None, extract_frame_local, video_path, ts, frame_path)
            if not ok:
                continue

            # ── Score check ───────────────────────────────────────────────
            score = await quick_score_check(frame_path)
            cur_t, cur_r = score.get("titans"), score.get("rival")
            cur_clock = score.get("clock")
            cur_q = score.get("quarter")

            if cur_t is not None or cur_r is not None:
                await manager.send(sid, {
                    "type": "score_update",
                    "titans_score": cur_t, "rival_score": cur_r,
                    "quarter": cur_q, "clock": cur_clock,
                    "video_ts": f"{ts//60}:{ts%60:02d}",
                })

            # ── Quarter change → scan for stats screen ────────────────────
            if cur_q and cur_q != last_quarter and last_quarter is not None:
                qkey = f"{last_quarter}→{cur_q}"
                if qkey not in quarter_break_scanned:
                    quarter_break_scanned.add(qkey)
                    await status(f"🏁 Quarter break: {last_quarter} → {cur_q} — scanning for stats screen...", "info")
                    qb_frame = os.path.join(work_dir, "qbreak_frame.jpg")
                    # Scan 8 frames over the 30 seconds BEFORE the quarter change
                    for qts in range(max(0, ts - 30), ts + 10, 5):
                        ok_q = await loop.run_in_executor(None, extract_frame_local, video_path, qts, qb_frame)
                        if ok_q:
                            screen = await analyze_timeout_screen(qb_frame, players)
                            if screen.get("has_stats") and screen.get("player_stats"):
                                pstats = screen["player_stats"]
                                label = "🏁 Halftime stats" if "2" in str(last_quarter) else f"📊 Q{last_quarter} break stats"
                                await status(f"{label} found — {len(pstats)} players!", "success")
                                await manager.send(sid, {
                                    "type": "timeout_stats",
                                    "video_ts": f"{ts//60}:{ts%60:02d}",
                                    "player_stats": pstats,
                                    "is_halftime": "2" in str(last_quarter),
                                })
                                break
            last_quarter = cur_q or last_quarter

            # ── Timeout detection (clock unchanged for ~30s) ──────────────
            if cur_clock and cur_clock == last_clock:
                clock_same_count += 1
                if clock_same_count == 10 and ts not in timeout_frames_analyzed:
                    timeout_frames_analyzed.add(ts)
                    await status(f"⏸ Stoppage @{ts//60}:{ts%60:02d} — checking stats overlay...", "info")
                    screen = await analyze_timeout_screen(frame_path, players)
                    if screen.get("has_stats") and screen.get("player_stats"):
                        pstats = screen["player_stats"]
                        await status(f"📊 Stats screen found! {len(pstats)} players", "success")
                        await manager.send(sid, {
                            "type": "timeout_stats",
                            "video_ts": f"{ts//60}:{ts%60:02d}",
                            "player_stats": pstats,
                            "is_halftime": screen.get("is_halftime", False),
                        })
            else:
                clock_same_count = 0
            last_clock = cur_clock

            # ── Score change detection ────────────────────────────────────
            titans_delta, rival_delta = 0, 0
            if last_score["titans"] is not None and cur_t is not None:
                d = cur_t - last_score["titans"]
                titans_delta = d if 0 < d <= 4 else 0
            if last_score["rival"] is not None and cur_r is not None:
                d = cur_r - last_score["rival"]
                rival_delta = d if 0 < d <= 4 else 0
            if cur_t is not None: last_score["titans"] = cur_t
            if cur_r is not None: last_score["rival"]  = cur_r

            if titans_delta > 0 or rival_delta > 0:
                play_ts  = max(60, ts - score_interval * 2)
                play_end = ts + score_interval
                play_fmt = f"{play_ts//60}:{play_ts%60:02d}"
                score_delta_cv = {"titans": titans_delta, "rival": rival_delta}
                await status(f"⚡ @{play_fmt}: Titans+{titans_delta} Rival+{rival_delta} — analyzing play...")

                # ── Layer 1: CV Tracking — YOLO preferred, OpenCV fallback ─
                cv_result = None
                if YOLO_AVAILABLE:
                    cv_result = await loop.run_in_executor(
                        None, analyze_clip_yolo,
                        video_path, float(play_ts), float(play_end),
                        score_delta_cv, titans_color, rival_color,
                    )
                if cv_result is None:
                    cv_result = await loop.run_in_executor(
                        None, cv_analyze_clip,
                        video_path, float(play_ts), float(play_end),
                        score_delta_cv, titans_color, rival_color,
                    )
                cv_scorer  = cv_result.get("scorer")
                cv_assist  = cv_result.get("assister")
                # STL removed from tracked stats
                cv_tracks  = cv_result.get("player_tracks", {})

                # Map CV track IDs → player names via jersey learning
                # (track_id from warmup scan → learned_jerseys entry)
                def tid_to_player(tid):
                    return warmup_track_to_player.get(tid)

                if cv_scorer and cv_scorer.get("confidence", 0) >= 0.60:
                    cv_player = tid_to_player(cv_scorer["track_id"])
                    cv_team   = cv_scorer.get("team", "titans")
                    cv_stype  = cv_scorer.get("shot_type", "2PT")
                    cv_zone   = cv_scorer.get("zone", "")
                    cv_stat   = {"2PT": "2PT_MADE", "3PT": "3PT_MADE", "FT": "FT_MADE"}.get(cv_stype, "2PT_MADE")
                    if cv_player and cv_team == "titans":
                        await emit_event(
                            "cv_tracking", play_fmt,
                            cv_player, "titans", cv_stat,
                            cv_scorer["confidence"],
                            f"CV tracking: {cv_zone} | {cv_result.get('summary','')}",
                        )
                    if cv_assist and cv_assist.get("confidence", 0) >= 0.50:
                        assist_player = tid_to_player(cv_assist["track_id"])
                        if assist_player:
                            await emit_event("cv_tracking", play_fmt, assist_player, "titans", "AST",
                                             cv_assist["confidence"], f"CV assist: passed to scorer")

                # ── Layer 2: Extract 8 frames from the play window ────────
                play_tss = [play_ts + int(j * (score_interval * 2) / 7) for j in range(8)]
                play_frames_data = await loop.run_in_executor(
                    None, extract_frames_local_batch, video_path, play_tss, work_dir
                )
                fps = [p for _, p in play_frames_data]

                if fps:
                    # ── EasyOCR pass (no-cost jersey OCR) ─────────────────
                    ocr_numbers = []
                    for fp in fps:
                        ocr_numbers.extend(extract_jersey_numbers_ocr(fp))

                    # ── Copa Talento Vision analysis with crop+zoom ────────
                    score_before = {"titans": (cur_t or 0) - titans_delta, "rival": (cur_r or 0) - rival_delta}
                    score_after  = {"titans": cur_t or 0, "rival": cur_r or 0}
                    events, new_jerseys, play_desc = await analyze_play_sequence(
                        fps, players, learned_jerseys, score_before, score_after, play_fmt,
                        player_profiles=player_profiles,
                        titans_jersey_color=titans_color,
                        rival_jersey_color=rival_color,
                    )

                    if new_jerseys:
                        learned_jerseys.update(new_jerseys)
                        await manager.send(sid, {"type": "jersey_update", "map": learned_jerseys})

                    # Merge OCR + Claude jersey discoveries
                    for ev in events:
                        pname = ev.get("player", "")
                        if pname and pname not in ("UNKNOWN_TITANS", "RIVAL") and ocr_numbers:
                            best_ocr = max(ocr_numbers, key=lambda x: x["confidence"])
                            num = best_ocr["number"]
                            if num not in learned_jerseys and best_ocr["confidence"] >= 0.6:
                                learned_jerseys[num] = pname

                    for ev in events:
                        await emit_event(
                            "full_auto", play_fmt,
                            ev.get("player", ""), ev.get("team", "titans"),
                            ev.get("stat", ""), ev.get("confidence", 0),
                            ev.get("reasoning", play_desc or ""),
                        )
                    if play_desc:
                        await status(f"🎬 {play_fmt}: {play_desc}", "success")

                    for fp in fps:
                        try: os.remove(fp)
                        except: pass

            # ── Check for whistle event near this timestamp ────────────────
            # Only if no score change (fouls w/o free throws, turnovers, etc.)
            if titans_delta == 0 and rival_delta == 0:
                nearby_whistles = [
                    wts for wts in whistle_set
                    if whistle_set[wts] == "whistle" and abs(wts - ts) <= score_interval
                    and wts not in analyzed_whistles
                ]
                nearby_whistle = len(nearby_whistles) > 0
            else:
                nearby_whistle = False

            if nearby_whistle:
                whistle_ts = min(nearby_whistles, key=lambda wts: abs(wts - ts))
                analyzed_whistles.add(whistle_ts)
                foul_fmt = f"{whistle_ts//60}:{whistle_ts%60:02d}"

                # Extract frames around the whistle for referee signal analysis
                foul_tss = [max(0, whistle_ts - 1) + j for j in range(8)]
                foul_frames = await loop.run_in_executor(
                    None, extract_frames_local_batch, video_path, foul_tss, work_dir
                )
                ffps = [p for _, p in foul_frames]

                if ffps:
                    await status(f"🚨 Whistle @{foul_fmt} — reading referee signal...", "info")
                    foul_result = await analyze_referee_foul_sequence(
                        ffps, players, learned_jerseys, foul_fmt
                    )
                    if foul_result.get("foul_called") and foul_result.get("confidence", 0) >= 0.5:
                        player_name = (foul_result.get("player_name") or
                                       learned_jerseys.get(str(foul_result.get("jersey_number", ""))) or
                                       "Jugador desconocido")
                        team = foul_result.get("team", "titans")
                        await emit_event(
                            "whistle_vision", foul_fmt,
                            player_name, team, "FOUL",
                            foul_result.get("confidence", 0),
                            f"Referee signal: {foul_result.get('reasoning', '')} | Type: {foul_result.get('foul_type', '?')}",
                        )
                        jn = foul_result.get("jersey_number")
                        if jn and str(jn) not in learned_jerseys and foul_result.get("player_name"):
                            learned_jerseys[str(jn)] = foul_result["player_name"]
                            await manager.send(sid, {"type": "jersey_update", "map": learned_jerseys})

                    # ── Substitution detection at this whistle ────────────
                    current_on_court = list(on_court_log.keys())
                    sub_result = await detect_substitution(ffps, players, learned_jerseys, current_on_court)
                    if sub_result.get("substitution_detected") and sub_result.get("confidence", 0) >= 0.55:
                        sub_in  = sub_result.get("sub_in")
                        sub_out = sub_result.get("sub_out")
                        jn_in   = str(sub_result.get("jersey_in", "") or "")
                        jn_out  = str(sub_result.get("jersey_out", "") or "")

                        # Update jersey map from sub frames (best jersey-read opportunity)
                        for jn, pname in [(jn_in, sub_in), (jn_out, sub_out)]:
                            if jn and pname and pname in players and jn not in learned_jerseys:
                                learned_jerseys[jn] = pname
                                await manager.send(sid, {"type": "jersey_update", "map": learned_jerseys})

                        # Track minutes: record sub_out leaving, sub_in entering
                        if sub_out and sub_out in on_court_log:
                            secs_played = whistle_ts - on_court_since.get(sub_out, 0)
                            total_minutes[sub_out] = total_minutes.get(sub_out, 0) + secs_played
                            del on_court_log[sub_out]
                            del on_court_since[sub_out]
                        if sub_in and sub_in in players:
                            on_court_log[sub_in]  = True
                            on_court_since[sub_in] = whistle_ts

                        await manager.send(sid, {
                            "type": "substitution",
                            "video_ts": foul_fmt,
                            "sub_in":   sub_in,
                            "sub_out":  sub_out,
                            "jersey_in":  jn_in,
                            "jersey_out": jn_out,
                            "confidence": sub_result.get("confidence", 0),
                        })
                        if sub_in or sub_out:
                            await status(f"🔄 Sub @{foul_fmt}: {sub_out or '?'} OUT → {sub_in or '?'} IN", "info")

                        # Also update minutes totals and emit to frontend
                        mins_update = {p: round(total_minutes.get(p, 0) / 60, 1) for p in players}
                        await manager.send(sid, {"type": "minutes_update", "minutes": mins_update})

                    for fp in ffps:
                        try: os.remove(fp)
                        except: pass

            pct = int((i + 1) / total * 100)
            await manager.send(sid, {"type": "auto_progress", "pct": pct, "phase": "scanning"})

        # ─── PHASE 4: Whisper commentary (if available) ──────────────────
        if whisper_task:
            await status("🎙 Checking Whisper transcription results...", "info")
            segments = await whisper_task
            if segments:
                await status(f"✓ Commentary: {len(segments)} segments — extracting player events...", "success")
                audio_events = await analyze_transcription_chunks(segments, players)
                for ev in audio_events:
                    await emit_event(
                        "audio", ev.get("video_ts", ""),
                        ev.get("player", ""), ev.get("team", ""),
                        ev.get("stat", ""), ev.get("confidence", 0),
                        ev.get("quote", ""),
                    )
                await status(f"✓ Commentary analysis: {len(audio_events)} events", "success")
            else:
                await status("ℹ No commentary detected — video-only analysis", "info")

        # ── Finalize minutes for players still on court at game end ───────
        for p, since in on_court_since.items():
            total_minutes[p] = total_minutes.get(p, 0) + (duration - since)
        final_mins = {p: round(total_minutes.get(p, 0) / 60, 1) for p in players}
        await manager.send(sid, {"type": "minutes_update", "minutes": final_mins})

        await status("✅ Full Auto complete! All plays detected and streamed.", "success")
        await manager.send(sid, {"type": "auto_done", "learned_jerseys": learned_jerseys, "minutes": final_mins})

    except asyncio.CancelledError:
        await status("Full Auto stopped.", "warn")
    except Exception as e:
        await status(f"❌ Full Auto error: {e}", "error")
        print(f"full_auto_pipeline error: {e}", flush=True)
        import traceback; traceback.print_exc()
    finally:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass
        active_tasks.pop(f"auto_{sid}", None)
