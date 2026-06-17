import os, uuid, asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from youtube_utils import get_transcript_chunks, get_video_info, get_live_transcript_since, extract_video_id
from claude_analyzer import analyze_chunk

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
