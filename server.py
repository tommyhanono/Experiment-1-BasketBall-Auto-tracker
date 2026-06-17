import os, uuid, asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from youtube_utils import get_transcript_chunks, get_video_info, get_live_transcript_since, extract_video_id, extract_frame_at
from claude_analyzer import analyze_chunk, analyze_frame_for_stats

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
