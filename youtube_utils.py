import asyncio
import re
import os
import subprocess
import tempfile
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
        api = YouTubeTranscriptApi()
        try:
            result = api.fetch(video_id, languages=["es", "en"])
            return list(result)
        except (NoTranscriptFound, TranscriptsDisabled):
            try:
                transcript_list = api.list(video_id)
                t = transcript_list.find_generated_transcript(["es", "en"])
                return list(t.fetch())
            except Exception:
                return []
        except Exception:
            try:
                result = api.fetch(video_id)
                return list(result)
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
        # Handle both dict and FetchedTranscriptSnippet objects
        if hasattr(seg, 'start'):
            start = seg.start
            text = seg.text.strip() if hasattr(seg, 'text') else ''
        else:
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
            api = YouTubeTranscriptApi()
            result = api.fetch(video_id, languages=["es", "en"])
            def seg_start(s):
                return s.start if hasattr(s, 'start') else s.get("start", 0)
            return [s for s in result if seg_start(s) > since_seconds]
        except Exception:
            return []

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch)


def _find_ffmpeg() -> tuple[str, str]:
    """Return (ffmpeg_binary_path, directory_for_yt_dlp_flag)."""
    import shutil
    # 1. System PATH
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg, os.path.dirname(sys_ffmpeg)
    # 2. imageio bundled binary — ensure a symlink named "ffmpeg" exists
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            ffdir = os.path.dirname(exe)
            fflink = os.path.join(ffdir, "ffmpeg")
            if not os.path.exists(fflink):
                os.symlink(exe, fflink)
            return fflink, ffdir
    except Exception:
        pass
    # 3. ~/bin
    home = os.path.expanduser("~/bin/ffmpeg")
    if os.path.exists(home):
        return home, os.path.dirname(home)
    raise RuntimeError("ffmpeg not found. Run: pip install imageio[ffmpeg]")


async def extract_frame_at(url: str, timestamp: int, frames_dir: str) -> str | None:
    """Download a 4-second clip at `timestamp` and extract one JPEG frame."""
    os.makedirs(frames_dir, exist_ok=True)
    clip_path = os.path.join(frames_dir, f"clip_{timestamp}.mp4")
    frame_path = os.path.join(frames_dir, f"frame_{timestamp}.jpg")

    try:
        ffmpeg_bin, ffmpeg_dir = _find_ffmpeg()
    except RuntimeError as e:
        print(f"extract_frame_at: {e}")
        return None

    # Download a 4-second segment with yt-dlp at low quality
    dl_cmd = [
        "yt-dlp",
        "--ffmpeg-location", ffmpeg_dir,
        "-f", "best[height<=480][ext=mp4]/best[height<=480]/best",
        "--download-sections", f"*{timestamp}-{timestamp+4}",
        "--no-playlist", "-q", "--no-warnings",
        "-o", clip_path, url,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *dl_cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        await asyncio.wait_for(proc.wait(), timeout=60)
    except (asyncio.TimeoutError, Exception) as e:
        print(f"yt-dlp error: {e}")
        return None

    if not os.path.exists(clip_path):
        return None

    # Extract 1 frame with ffmpeg
    ff_cmd = [ffmpeg_bin, "-y", "-i", clip_path, "-vframes", "1", "-q:v", "3", frame_path]
    try:
        proc2 = await asyncio.create_subprocess_exec(
            *ff_cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        await asyncio.wait_for(proc2.wait(), timeout=15)
    except (asyncio.TimeoutError, Exception) as e:
        print(f"ffmpeg error: {e}")
        return None
    finally:
        try:
            os.remove(clip_path)
        except Exception:
            pass

    return frame_path if os.path.exists(frame_path) else None
