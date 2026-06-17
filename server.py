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
    transcribe_audio,
)
from claude_analyzer import (
    analyze_chunk, analyze_frame_for_stats, analyze_play_sequence,
    quick_score_check, analyze_transcription_chunks,
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
    score_interval: int = 5    # seconds between local score checks


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
        full_auto_pipeline(req.url, req.session_id, req.players, req.jersey_map, req.score_interval)
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


async def full_auto_pipeline(url: str, sid: str, players: list[str], jersey_map: dict, score_interval: int):
    """
    The REAL fully automatic pipeline:
    1. Try Whisper audio transcription (gets individual player names from commentary)
    2. Download video at 360p for local frame analysis
    3. Scan scoreboard every `score_interval` seconds using local ffmpeg (FAST, no yt-dlp per frame)
    4. On score change → extract 6 frames → Claude Vision play analysis (jersey number ID)
    5. All events streamed in real-time via WebSocket
    """
    import shutil

    work_dir = f"/tmp/auto_{sid}"
    os.makedirs(work_dir, exist_ok=True)
    video_path = os.path.join(work_dir, "game.mp4")
    learned_jerseys = dict(jersey_map)
    last_score = {"titans": None, "rival": None}

    async def status(msg, level="info"):
        await manager.send(sid, {"type": "status", "msg": msg, "level": level})

    async def progress(pct, phase=""):
        await manager.send(sid, {"type": "auto_progress", "pct": pct, "phase": phase})

    try:
        # ── Phase 1: Audio transcription (parallel with download) ─────────
        await status("🎙 Checking audio for commentary...")
        audio_task = asyncio.create_task(
            transcribe_audio(url, work_dir,
                progress_cb=lambda m: manager.send(sid, {"type": "status", "msg": f"🎙 {m}", "level": "info"}))
        )

        # ── Phase 2: Download full video ─────────────────────────────────
        await status("⬇ Downloading game video (this takes a few minutes)...")
        await manager.send(sid, {"type": "auto_phase", "phase": "download"})

        download_ok = await download_video_local(url, video_path,
            progress_cb=lambda m: manager.send(sid, {"type": "status", "msg": f"⬇ {m}", "level": "info"}))

        if not download_ok:
            await status("❌ Could not download video. Check URL.", "error")
            return

        info = await get_video_info(url)
        duration = int(info.get("duration", 0)) if info else 0
        if not duration:
            await status("❌ Could not read video duration.", "error")
            return

        await status(f"✓ Video downloaded ({os.path.getsize(video_path)//1048576}MB, {duration//60}m). Scanning for plays...")
        await manager.send(sid, {"type": "auto_phase", "phase": "scanning"})

        # ── Phase 3: Score scan via local frames ──────────────────────────
        timestamps = list(range(60, min(duration, 7200), score_interval))
        total = len(timestamps)
        frame_path = os.path.join(work_dir, "current_frame.jpg")

        for i, ts in enumerate(timestamps):
            if f"auto_{sid}" not in active_tasks:
                break

            # Extract frame locally (fast: ~0.1s per frame)
            loop = asyncio.get_event_loop()
            ok = await loop.run_in_executor(None, extract_frame_local, video_path, ts, frame_path)

            if ok:
                score = await quick_score_check(frame_path)
                cur_t = score.get("titans")
                cur_r = score.get("rival")

                if cur_t is not None or cur_r is not None:
                    await manager.send(sid, {
                        "type": "score_update",
                        "titans_score": cur_t,
                        "rival_score": cur_r,
                        "quarter": score.get("quarter"),
                        "clock": score.get("clock"),
                        "video_ts": f"{ts//60}:{ts%60:02d}",
                    })

                titans_delta = 0
                rival_delta  = 0
                if last_score["titans"] is not None and cur_t is not None:
                    raw_delta = cur_t - last_score["titans"]
                    titans_delta = raw_delta if 0 < raw_delta <= 4 else 0
                if last_score["rival"] is not None and cur_r is not None:
                    raw_delta = cur_r - last_score["rival"]
                    rival_delta = raw_delta if 0 < raw_delta <= 4 else 0

                if cur_t is not None:
                    last_score["titans"] = cur_t
                if cur_r is not None:
                    last_score["rival"] = cur_r

                # ── Score changed → analyze the play (validate delta is sensible) ──
                # Valid basketball increments: 1 (FT), 2 (field goal), 3 (three-pointer)
                # Anything > 4 in one 5-second window is likely a bad read
                titans_delta = min(titans_delta, 4)
                rival_delta  = min(rival_delta, 4)
                if titans_delta > 0 or rival_delta > 0:
                    play_ts = max(60, ts - score_interval)
                    play_fmt = f"{play_ts//60}:{play_ts%60:02d}"

                    await status(f"⚡ Score change @{play_fmt}: Titans+{titans_delta} Rival+{rival_delta} — analyzing play...")

                    # Extract 6 frames from the play window (local, instant)
                    play_timestamps = [play_ts + int(j * score_interval / 5) for j in range(6)]
                    play_frames = await loop.run_in_executor(
                        None, extract_frames_local_batch, video_path, play_timestamps, work_dir
                    )
                    frame_paths = [p for _, p in play_frames]

                    if frame_paths:
                        score_before = {
                            "titans": (cur_t or 0) - titans_delta,
                            "rival":  (cur_r or 0) - rival_delta,
                        }
                        score_after = {"titans": cur_t or 0, "rival": cur_r or 0}

                        events, new_jerseys, play_desc = await analyze_play_sequence(
                            frame_paths, players, learned_jerseys, score_before, score_after, play_fmt
                        )

                        if new_jerseys:
                            learned_jerseys.update(new_jerseys)
                            await manager.send(sid, {"type": "jersey_update", "map": learned_jerseys})

                        for ev in events:
                            player = ev.get("player", "UNKNOWN")
                            if player == "UNKNOWN_TITANS":
                                player = "Titans (sin identificar)"
                            await manager.send(sid, {
                                "type": "ai_event",
                                "id": str(uuid.uuid4())[:8],
                                "source": "full_auto",
                                "video_ts": play_fmt,
                                "player": player,
                                "team": ev.get("team", "titans"),
                                "stat": ev.get("stat", ""),
                                "confidence": ev.get("confidence", 0),
                                "quote": ev.get("reasoning", play_desc or ""),
                            })

                        if play_desc:
                            await status(f"🎬 {play_fmt}: {play_desc}", "success")

                        # Clean up play frames
                        for fp in frame_paths:
                            try:
                                os.remove(fp)
                            except Exception:
                                pass

            # Update progress bar
            pct = int((i + 1) / total * 100)
            await manager.send(sid, {"type": "auto_progress", "pct": pct, "phase": "scanning"})

        # ── Phase 4: Audio results (if Whisper finished) ──────────────────
        await status("🎙 Processing audio transcription results...")
        audio_segments = await audio_task

        if audio_segments:
            await status(f"✓ Audio: {len(audio_segments)} spoken segments — extracting stats with Claude...")
            await manager.send(sid, {"type": "auto_phase", "phase": "audio"})
            audio_events = await analyze_transcription_chunks(audio_segments, players)
            for ev in audio_events:
                await manager.send(sid, {
                    "type": "ai_event",
                    "id": str(uuid.uuid4())[:8],
                    "source": "audio",
                    "video_ts": ev.get("video_ts", ""),
                    "player": ev.get("player", ""),
                    "team": ev.get("team", ""),
                    "stat": ev.get("stat", ""),
                    "confidence": ev.get("confidence", 0),
                    "quote": ev.get("quote", ""),
                })
            await status(f"✓ Audio analysis: {len(audio_events)} events from commentary", "success")
        else:
            await status("ℹ No commentary audio detected — visual analysis only", "info")

        await status("✅ Full Auto complete! Review events in the feed.", "success")
        await manager.send(sid, {"type": "auto_done", "learned_jerseys": learned_jerseys})

    except asyncio.CancelledError:
        await status("Full Auto stopped.", "warn")
    except Exception as e:
        await status(f"❌ Full Auto error: {e}", "error")
        print(f"full_auto_pipeline error: {e}")
    finally:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass
        active_tasks.pop(f"auto_{sid}", None)
