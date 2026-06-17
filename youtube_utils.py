import asyncio
import re
import os
import subprocess
import tempfile
import base64
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


async def extract_clip_frames(url: str, timestamp: int, frames_dir: str, num_frames: int = 6) -> list[str]:
    """Download a 6-second clip and extract `num_frames` evenly spaced JPEGs from it."""
    os.makedirs(frames_dir, exist_ok=True)

    try:
        ffmpeg_bin, ffmpeg_dir = _find_ffmpeg()
    except RuntimeError as e:
        print(f"extract_clip_frames: {e}")
        return []

    # Start 2 seconds before the timestamp to catch the play leading up to it
    start = max(0, timestamp - 2)
    clip_path = os.path.join(frames_dir, f"clip_seq_{timestamp}.mp4")

    dl_cmd = [
        "yt-dlp",
        "--ffmpeg-location", ffmpeg_dir,
        "-f", "best[height<=480][ext=mp4]/best[height<=480]/best",
        "--download-sections", f"*{start}-{start+6}",
        "--no-playlist", "-q", "--no-warnings",
        "-o", clip_path, url,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *dl_cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        await asyncio.wait_for(proc.wait(), timeout=60)
    except (asyncio.TimeoutError, Exception) as e:
        print(f"clip download error: {e}")
        return []

    if not os.path.exists(clip_path):
        return []

    # Extract num_frames frames evenly spaced across the clip using fps filter
    pattern = os.path.join(frames_dir, f"seq_{timestamp}_%02d.jpg")
    fps = num_frames / 6.0  # spread num_frames across 6 seconds
    ff_cmd = [
        ffmpeg_bin, "-y", "-i", clip_path,
        "-vf", f"fps={fps:.3f}", "-q:v", "3", pattern
    ]
    try:
        proc2 = await asyncio.create_subprocess_exec(
            *ff_cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        await asyncio.wait_for(proc2.wait(), timeout=20)
    except (asyncio.TimeoutError, Exception) as e:
        print(f"frame extraction error: {e}")
        return []
    finally:
        try:
            os.remove(clip_path)
        except Exception:
            pass

    # Collect the extracted frames
    frames = sorted(
        f for f in (os.path.join(frames_dir, fn) for fn in os.listdir(frames_dir)
                    if fn.startswith(f"seq_{timestamp}_") and fn.endswith(".jpg"))
        if os.path.exists(f)
    )
    return frames


# ── Full-Auto: download video once, analyze locally ────────────────────────

async def download_video_local(url: str, output_path: str, progress_cb=None) -> bool:
    """Download video at 360p quality for local frame analysis. Returns True on success."""
    try:
        ffmpeg_bin, ffmpeg_dir = _find_ffmpeg()
    except RuntimeError:
        return False

    dl_cmd = [
        "yt-dlp",
        "--ffmpeg-location", ffmpeg_dir,
        "-f", "best[height<=360][ext=mp4]/best[height<=360]/mp4/best[height<=480]",
        "--no-playlist", "--no-warnings",
        "-o", output_path, url,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *dl_cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        # Stream stderr to detect download progress
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            text = line.decode(errors='replace').strip()
            if progress_cb and '[download]' in text:
                await progress_cb(text)
        await asyncio.wait_for(proc.wait(), timeout=1800)  # 30 min max
    except (asyncio.TimeoutError, Exception) as e:
        print(f"download_video_local error: {e}")
        return False

    return os.path.exists(output_path) and os.path.getsize(output_path) > 100_000


def extract_frame_local(video_path: str, timestamp_sec: float, out_path: str) -> bool:
    """Extract one JPEG frame from a local video file at `timestamp_sec`. Synchronous."""
    try:
        ffmpeg_bin, _ = _find_ffmpeg()
    except RuntimeError:
        return False

    cmd = [
        ffmpeg_bin, "-y",
        "-ss", str(timestamp_sec),
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "3",
        out_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        return result.returncode == 0 and os.path.exists(out_path)
    except Exception:
        return False


def extract_frames_local_batch(video_path: str, timestamps: list[float], work_dir: str) -> list[tuple[float, str]]:
    """Extract multiple frames from a local video. Returns list of (timestamp, path)."""
    os.makedirs(work_dir, exist_ok=True)
    results = []
    for ts in timestamps:
        out_path = os.path.join(work_dir, f"f_{int(ts):06d}.jpg")
        if extract_frame_local(video_path, ts, out_path):
            results.append((ts, out_path))
    return results


async def transcribe_audio(url: str, work_dir: str, progress_cb=None) -> list[dict]:
    """Download audio and transcribe with faster-whisper. Returns [{start, end, text}]."""
    os.makedirs(work_dir, exist_ok=True)
    audio_path = os.path.join(work_dir, "audio.mp3")

    try:
        ffmpeg_bin, ffmpeg_dir = _find_ffmpeg()
    except RuntimeError:
        return []

    dl_cmd = [
        "yt-dlp",
        "--ffmpeg-location", ffmpeg_dir,
        "-x", "--audio-format", "mp3", "--audio-quality", "5",
        "--no-playlist", "-q", "-o", audio_path, url,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *dl_cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        await asyncio.wait_for(proc.wait(), timeout=900)
    except Exception as e:
        print(f"audio download error: {e}")
        return []

    if not os.path.exists(audio_path):
        return []

    if progress_cb:
        await progress_cb("Transcribing audio with Whisper...")

    def _transcribe():
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("small", device="cpu", compute_type="int8")
            segments, info = model.transcribe(
                audio_path, language="es", beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
            )
            result = [{"start": s.start, "end": s.end, "text": s.text.strip()}
                      for s in segments if s.text.strip() and len(s.text.strip()) > 3]
            return result, info.language, info.language_probability
        except ImportError:
            return [], "unknown", 0.0

    loop = asyncio.get_event_loop()
    segments, lang, prob = await loop.run_in_executor(None, _transcribe)

    try:
        os.remove(audio_path)
    except Exception:
        pass

    if progress_cb:
        await progress_cb(f"Transcription: {len(segments)} segments | {lang} {prob:.0%}")

    return segments


def extract_audio_from_video(video_path: str, audio_path: str) -> bool:
    """Extract mono audio from downloaded video file using ffmpeg. Returns True on success."""
    try:
        ffmpeg_bin, _ = _find_ffmpeg()
    except RuntimeError:
        return False
    cmd = [ffmpeg_bin, "-y", "-i", video_path, "-ac", "1", "-ar", "22050",
           "-q:a", "5", "-vn", audio_path]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    return result.returncode == 0 and os.path.exists(audio_path)


def detect_whistle_timestamps(audio_path: str) -> list[dict]:
    """
    Detect referee whistle timestamps using bandpass FIR energy method.
    Based on: 'Automated Referee Whistle Sound Detection for Extraction of Highlights
    from Sports Video' — achieves 92.5% sensitivity.

    Returns list of {"ts": float, "type": "whistle"|"cheer", "energy": float}.
    """
    import numpy as np

    try:
        ffmpeg_bin, _ = _find_ffmpeg()
    except RuntimeError:
        return []

    # Decode audio to raw PCM float32 at 22050 Hz via ffmpeg pipe
    cmd = [ffmpeg_bin, "-y", "-i", audio_path, "-ac", "1", "-ar", "22050",
           "-f", "f32le", "-"]
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0 or not result.stdout:
        return []

    data = np.frombuffer(result.stdout, dtype=np.float32)
    sr = 22050
    duration = len(data) / sr

    # ── Bandpass FIR filter (2800-4200 Hz) for whistle frequency range ──────
    try:
        from scipy.signal import firwin, lfilter
        nyq = sr / 2
        taps = firwin(65, [2800 / nyq, 4200 / nyq], pass_zero=False)
        whistle_signal = lfilter(taps, 1.0, data)
    except ImportError:
        # Fallback: manual FFT bandpass using numpy
        fft = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), 1 / sr)
        mask = (freqs >= 2800) & (freqs <= 4200)
        fft_filtered = fft * mask
        whistle_signal = np.fft.irfft(fft_filtered, len(data))

    # ── Full-band signal for crowd cheer detection ───────────────────────────
    # Cheers = sustained broad-band high-energy bursts after scoring plays
    full_signal = data

    # ── Short-time energy (50ms window, 25ms hop) ───────────────────────────
    win_samps = int(sr * 0.05)
    hop_samps = int(sr * 0.025)

    def ste(sig):
        n = (len(sig) - win_samps) // hop_samps
        return np.array([
            np.sqrt(np.mean(sig[i * hop_samps: i * hop_samps + win_samps] ** 2))
            for i in range(n)
        ])

    whistle_ste  = ste(whistle_signal)
    full_ste     = ste(full_signal)
    times = np.arange(len(whistle_ste)) * hop_samps / sr

    # ── Decision thresholds ──────────────────────────────────────────────────
    w_mean, w_std = whistle_ste.mean(), whistle_ste.std()
    f_mean, f_std = full_ste.mean(), full_ste.std()

    # Whistles: sharp spike in whistle band, duration < 1.5 seconds
    whistle_thresh = w_mean + 3.0 * w_std
    # Cheers: broad-band energy spike, duration > 2 seconds (crowd reacts to scores)
    cheer_thresh   = f_mean + 2.5 * f_std

    whistle_on = whistle_ste > whistle_thresh
    cheer_on   = (full_ste > cheer_thresh) & (whistle_ste < whistle_thresh * 0.6)

    def group_events(mask, min_gap_sec=1.5, min_dur_frames=2, max_dur_frames=60):
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return []
        events, grp = [], [idxs[0]]
        for idx in idxs[1:]:
            if idx - grp[-1] < min_gap_sec / 0.025:
                grp.append(idx)
            else:
                if min_dur_frames <= len(grp) <= max_dur_frames:
                    events.append(times[int(np.median(grp))])
                grp = [idx]
        if min_dur_frames <= len(grp) <= max_dur_frames:
            events.append(times[int(np.median(grp))])
        return events

    whistle_events = group_events(whistle_on, min_gap_sec=1.5, min_dur_frames=2, max_dur_frames=30)
    cheer_events   = group_events(cheer_on,   min_gap_sec=2.0, min_dur_frames=6, max_dur_frames=200)

    results = [{"ts": t, "type": "whistle"} for t in whistle_events]
    results += [{"ts": t, "type": "cheer"}  for t in cheer_events]
    results.sort(key=lambda x: x["ts"])

    return results
