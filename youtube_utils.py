import asyncio
import re
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled


def extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=|youtu\.be/|embed/)([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def fmt_seconds(secs: float) -> str:
    secs = int(secs)
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


async def get_video_info(url: str) -> dict | None:
    try:
        import yt_dlp
        opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        loop = asyncio.get_event_loop()
        def _fetch():
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(url, download=False)
        info = await loop.run_in_executor(None, _fetch)
        return {
            "title": info.get("title", ""),
            "duration": info.get("duration", 0),
            "is_live": info.get("is_live", False),
            "thumbnail": info.get("thumbnail", ""),
        }
    except Exception as e:
        print(f"Video info error: {e}")
        return None


async def get_transcript_chunks(url: str, chunk_minutes: int = 8) -> list[dict]:
    """Fetch transcript and group into time chunks for Claude analysis."""
    video_id = extract_video_id(url)
    if not video_id:
        return []

    def _fetch():
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["es", "en", "auto"])
            return transcript
        except (NoTranscriptFound, TranscriptsDisabled):
            try:
                transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
                t = transcripts.find_generated_transcript(["es", "en"])
                return t.fetch()
            except Exception:
                return []

    loop = asyncio.get_event_loop()
    segments = await loop.run_in_executor(None, _fetch)

    if not segments:
        return []

    # Group into chunk_minutes-minute windows
    chunk_secs = chunk_minutes * 60
    chunks = []
    current_chunk = []
    current_start = 0.0

    for seg in segments:
        start = seg.get("start", 0)
        text = seg.get("text", "").strip()
        if not text:
            continue

        if not current_chunk:
            current_start = start

        if start - current_start >= chunk_secs and current_chunk:
            chunks.append({
                "start": current_start,
                "end": start,
                "start_fmt": fmt_seconds(current_start),
                "end_fmt": fmt_seconds(start),
                "text": " ".join(current_chunk),
            })
            current_chunk = [text]
            current_start = start
        else:
            current_chunk.append(text)

    if current_chunk:
        chunks.append({
            "start": current_start,
            "end": current_start + chunk_secs,
            "start_fmt": fmt_seconds(current_start),
            "end_fmt": fmt_seconds(current_start + chunk_secs),
            "text": " ".join(current_chunk),
        })

    return chunks


async def get_live_transcript_since(url: str, since_seconds: float) -> list[dict]:
    """For live streams: fetch only new transcript segments since last check."""
    video_id = extract_video_id(url)
    if not video_id:
        return []

    def _fetch():
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["es", "en"])
            return [s for s in transcript if s.get("start", 0) > since_seconds]
        except Exception:
            return []

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch)
