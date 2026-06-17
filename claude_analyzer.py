import os, json, re, base64, asyncio, io
import anthropic
from PIL import Image

_client = None

def get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


STAT_KEYS = {
    "2PT_MADE": "2-point field goal made",
    "2PT_MISS": "2-point field goal missed/attempt only",
    "3PT_MADE": "3-point field goal made",
    "3PT_MISS": "3-point field goal missed/attempt only",
    "FT_MADE":  "free throw made",
    "FT_MISS":  "free throw missed/attempt only",
    "REB_OFF":  "offensive rebound",
    "REB_DEF":  "defensive rebound",
    "AST":      "assist",
    "TOV":      "turnover / lost ball",
    "STL":      "steal",
    "BLK":      "block",
    "FOUL":     "personal foul committed",
}

RIVAL_STATS = ["2PT_MADE", "2PT_MISS", "3PT_MADE", "3PT_MISS", "FT_MADE", "FT_MISS", "FOUL"]


async def analyze_chunk(text: str, timestamp: str, players: list[str]) -> list[dict]:
    """Analyze a transcript chunk and return detected basketball events."""
    if not text.strip():
        return []

    roster_str = "\n".join(f"- {p}" for p in players)
    stat_desc = "\n".join(f"  {k}: {v}" for k, v in STAT_KEYS.items())

    prompt = f"""You are a basketball statistics analyst. Analyze this game commentary transcript and extract individual player statistics events.

TITANS ROSTER (only track individual stats for these players):
{roster_str}

STAT KEYS to use:
{stat_desc}

For rival team events, use player name "RIVAL" — only track: 2PT_MADE, 2PT_MISS, 3PT_MADE, 3PT_MISS, FT_MADE, FT_MISS, FOUL.

Transcript segment (video timestamp {timestamp}):
{text}

Return ONLY valid JSON, no other text. Format:
{{
  "events": [
    {{
      "player": "exact player name from roster, or RIVAL",
      "team": "titans" or "rival",
      "stat": "one of the stat keys above",
      "confidence": 0.0 to 1.0,
      "quote": "the exact phrase that indicates this event"
    }}
  ]
}}

Rules:
- Only include events where a player name is explicitly mentioned OR clearly implied by context
- Use confidence < 0.6 for uncertain events
- Do not infer rebounds unless explicitly mentioned
- If no events found, return {{"events": []}}"""

    def _call():
        resp = get_client().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text

    loop = asyncio.get_event_loop()
    try:
        raw = await loop.run_in_executor(None, _call)
        raw = raw.strip()
        # Strip markdown code fences first
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
        # Find JSON object
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1:
            data = json.loads(raw[start:end+1])
            return data.get("events", [])
    except Exception as e:
        print(f"Analyzer error: {e}")
    return []


async def analyze_frame_for_stats(image_path: str, players: list[str], context: str) -> dict:
    """Analyze a video frame to extract scoreboard and any player graphics."""
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        roster_str = ", ".join(players)
        prompt = f"""You are analyzing a basketball broadcast frame. The tracked team is "Titans".

Titans roster: {roster_str}
Context: {context}

Extract ONLY what is clearly visible. Return ONLY valid JSON:
{{
  "titans_score": null or integer (Titans team score if visible),
  "rival_score": null or integer (opposing team score if visible),
  "rival_name": null or string (opponent team name if visible),
  "quarter": null or string (e.g. "Q1", "1st", "2nd half"),
  "clock": null or string (game clock if visible, e.g. "02:27"),
  "player_events": [],
  "text_on_screen": "brief description of any text or graphics visible"
}}

If Titans are not visible or score is unclear, use null. Do not guess."""

        def _call():
            resp = get_client().messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                        {"type": "text", "text": prompt}
                    ]
                }]
            )
            return resp.content[0].text

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, _call)
        raw = raw.strip()
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1:
            return json.loads(raw[start:end+1])
    except Exception as e:
        print(f"Frame analyzer error: {e}")
    return {}


async def analyze_play_sequence(
    frame_paths: list[str],
    players: list[str],
    jersey_map: dict,          # {"7": "Joseph Gabay", "11": "Aaron Breziner", ...}
    score_before: dict,        # {"titans": 10, "rival": 14}
    score_after: dict,         # {"titans": 12, "rival": 14}
    video_ts: str,
) -> list[dict]:
    """
    Send a 6-frame clip sequence to Claude Vision.
    Detects who scored/fouled by reading jersey numbers.
    Returns list of events like analyze_chunk.
    """
    if not frame_paths:
        return []

    titans_delta = (score_after.get("titans") or 0) - (score_before.get("titans") or 0)
    rival_delta  = (score_after.get("rival") or 0)  - (score_before.get("rival") or 0)

    # Build roster string with jersey numbers if known
    if jersey_map:
        roster_lines = []
        for p in players:
            num = next((n for n, name in jersey_map.items() if name == p), "?")
            roster_lines.append(f"  #{num} — {p}")
        roster_str = "\n".join(roster_lines)
    else:
        roster_str = "\n".join(f"  {p}" for p in players)

    score_desc = []
    if titans_delta > 0:
        score_desc.append(f"Titans scored +{titans_delta} (now {score_after['titans']})")
    if rival_delta > 0:
        score_desc.append(f"Rival scored +{rival_delta} (now {score_after['rival']})")
    if not score_desc:
        score_desc = ["Score unchanged — possible foul, timeout, or replay"]
    score_summary = " | ".join(score_desc)

    prompt = f"""You are watching {len(frame_paths)} consecutive frames from a basketball game broadcast (frames in order, ~1 second apart).

What happened in this play window:
{score_summary}
Video time: {video_ts}

Titans roster (with jersey numbers if known):
{roster_str}

Your job: identify every INDIVIDUAL event in these frames.

How to identify players:
1. Read jersey numbers on the back or front of jerseys — match to the roster above
2. Look at who has the ball, who is shooting, who is at the free throw line
3. If you see "FOUL ON #X" or any text overlay — read it exactly
4. Look at the scoreboard — team foul count changes tell you a foul happened
5. Free throw situation = foul happened earlier; the FT shooter = the fouled player

Events to detect and report:
- 2PT_MADE / 2PT_MISS (inside the 3-point arc)
- 3PT_MADE / 3PT_MISS (behind the 3-point arc)
- FT_MADE / FT_MISS (free throw line)
- FOUL (player who committed the foul)
- REB_OFF / REB_DEF (if clearly visible)
- BLK / STL (if clearly visible)

Return ONLY valid JSON:
{{
  "events": [
    {{
      "player": "exact player name from roster, or RIVAL",
      "team": "titans" or "rival",
      "stat": "one stat key from the list above",
      "confidence": 0.0 to 1.0,
      "reasoning": "jersey #X visible | text overlay says | player at FT line | etc."
    }}
  ],
  "jersey_numbers_seen": {{"7": "visible on player at basket"}},
  "play_description": "one sentence describing what happened"
}}

Confidence guide:
- 0.9+ : jersey number clearly readable AND matches a known play
- 0.7-0.9 : jersey partially visible OR play type inferred from position
- 0.5-0.7 : educated guess from context
- below 0.5 : do not include the event

If you cannot identify the specific player with ≥0.5 confidence, use "UNKNOWN_TITANS" or "RIVAL"."""

    try:
        content = []
        for path in frame_paths:
            with open(path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": img_data}
            })
        content.append({"type": "text", "text": prompt})

        def _call():
            return get_client().messages.create(
                model="claude-sonnet-4-6",
                max_tokens=800,
                messages=[{"role": "user", "content": content}]
            ).content[0].text

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, _call)
        raw = raw.strip()
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
        s = raw.find('{')
        e = raw.rfind('}')
        if s != -1 and e != -1:
            data = json.loads(raw[s:e+1])
            return data.get("events", []), data.get("jersey_numbers_seen", {}), data.get("play_description", "")
    except Exception as err:
        print(f"Play sequence error: {err}")
    return [], {}, ""


def _compress_frame(path: str, max_dim: int = 480) -> bytes:
    """Resize and compress a frame to reduce token usage for score-only checks."""
    try:
        img = Image.open(path)
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=60)
        return buf.getvalue()
    except Exception:
        with open(path, "rb") as f:
            return f.read()


async def quick_score_check(frame_path: str) -> dict:
    """
    Fast, cheap score check — sends a compressed frame, asks only for numbers.
    Returns {"titans": int|None, "rival": int|None, "quarter": str|None, "clock": str|None}.
    """
    try:
        img_bytes = await asyncio.get_event_loop().run_in_executor(
            None, _compress_frame, frame_path
        )
        img_data = base64.b64encode(img_bytes).decode()

        prompt = """Basketball scoreboard frame. Extract ONLY the numbers from the scoreboard.
Return ONLY JSON, no text:
{"home":null,"away":null,"home_name":null,"away_name":null,"quarter":null,"clock":null}
Use null if not visible. Numbers only, no extra text."""

        def _call():
            return get_client().messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=80,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_data}},
                        {"type": "text", "text": prompt},
                    ]
                }]
            ).content[0].text

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, _call)
        raw = re.sub(r"```(?:json)?\s*", "", raw.strip()).strip()
        s = raw.find('{')
        e = raw.rfind('}')
        if s != -1 and e != -1:
            d = json.loads(raw[s:e+1])
            home_name = (d.get("home_name") or "").upper()
            away_name = (d.get("away_name") or "").upper()
            # Identify which side is Titans
            titans_is_home = "TITAN" in home_name
            titans_is_away = "TITAN" in away_name
            if titans_is_home:
                titans_score, rival_score = d.get("home"), d.get("away")
                rival_name = d.get("away_name")
            elif titans_is_away:
                titans_score, rival_score = d.get("away"), d.get("home")
                rival_name = d.get("home_name")
            else:
                # Default: assume home=Titans (Copa Talento usually shows Titans on left)
                titans_score, rival_score = d.get("home"), d.get("away")
                rival_name = d.get("away_name")
            return {
                "titans": int(titans_score) if titans_score is not None else None,
                "rival":  int(rival_score)  if rival_score  is not None else None,
                "rival_name": rival_name,
                "quarter": d.get("quarter"),
                "clock": d.get("clock"),
            }
    except Exception as e:
        print(f"quick_score_check: {e}")
    return {"titans": None, "rival": None, "quarter": None, "clock": None}


async def analyze_transcription_chunks(segments: list[dict], players: list[str]) -> list[dict]:
    """
    Break Whisper transcription segments into 5-minute chunks and analyze each with Claude.
    Returns flat list of all events.
    """
    if not segments:
        return []

    chunk_secs = 5 * 60
    chunks = []
    current_texts = []
    current_start = segments[0]["start"]

    for seg in segments:
        if seg["start"] - current_start >= chunk_secs and current_texts:
            chunks.append({"start": current_start, "text": " ".join(current_texts)})
            current_texts = [seg["text"]]
            current_start = seg["start"]
        else:
            current_texts.append(seg["text"])

    if current_texts:
        chunks.append({"start": current_start, "text": " ".join(current_texts)})

    all_events = []
    for chunk in chunks:
        ts_fmt = f"{int(chunk['start'])//60}:{int(chunk['start'])%60:02d}"
        events = await analyze_chunk(chunk["text"], ts_fmt, players)
        all_events.extend(events)
        await asyncio.sleep(0.1)

    return all_events


# ── EasyOCR jersey number detection ─────────────────────────────────────────

_ocr_reader = None

def _get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        except ImportError:
            pass
    return _ocr_reader


def extract_jersey_numbers_ocr(frame_path: str) -> list[dict]:
    """
    Use EasyOCR to find jersey numbers in a frame.
    Returns list of {"number": "7", "confidence": 0.85, "bbox": [x1,y1,x2,y2]}.
    Filters for 1-2 digit numbers (valid jersey numbers: 0-99).
    """
    reader = _get_ocr()
    if not reader:
        return []

    try:
        results = reader.readtext(frame_path, detail=1, paragraph=False)
        jerseys = []
        for (bbox, text, conf) in results:
            text = text.strip().lstrip('#').strip()
            if text.isdigit() and 0 <= int(text) <= 99 and conf >= 0.4:
                # bbox is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                jerseys.append({
                    "number": text,
                    "confidence": round(conf, 2),
                    "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                })
        return jerseys
    except Exception as e:
        print(f"OCR error: {e}")
        return []


# ── Referee foul signal analysis ─────────────────────────────────────────────

async def analyze_referee_foul_sequence(
    frame_paths: list[str],
    players: list[str],
    jersey_map: dict,
    video_ts: str,
) -> dict:
    """
    Analyze frames around a referee whistle to detect foul and jersey number.

    FIBA referee foul reporting sequence:
    1. Blow whistle + raise closed fist = foul called
    2. Point toward fouling player
    3. Show jersey number with fingers (1 hand for 0-9, 2 hands for 10+)
    4. Show foul type gesture (blocking = hands on hips, charging = fist punch, etc.)
    5. Show accumulation (raise 1-5 fingers = personal foul count)

    Returns dict with foul_called, jersey_number, player_name, foul_type, confidence.
    """
    if not frame_paths:
        return {"foul_called": False}

    if jersey_map:
        roster_lines = "\n".join(f"  #{num} → {name}" for num, name in sorted(jersey_map.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 99))
        if not roster_lines:
            roster_lines = "  (jersey numbers not yet configured)"
    else:
        roster_lines = "  (jersey numbers not yet configured)"

    prompt = f"""You are watching {len(frame_paths)} consecutive frames from a basketball broadcast IMMEDIATELY AFTER a referee whistle was detected in the audio.

Known jersey numbers → players:
{roster_lines}

All other Titans players (jersey unknown): {', '.join(p for p in players if p not in jersey_map.values())}

TASK: Identify if a foul was called and WHO committed it.

FIBA REFEREE FOUL SEQUENCE (look for this in frames):
1. FOUL CALLED signal: referee raises CLOSED FIST in the air
2. POINTING: referee extends arm and points toward the fouling player
3. JERSEY NUMBER SIGNAL (most important!):
   - 1 finger up = #1
   - 2 fingers = #2
   - ... up to 5 fingers = #5
   - Open hand (5 fingers) + then 1 finger = #6 (or shown as one hand 5 + other hand 1)
   - For numbers 0-9: shown with one hand
   - For numbers 10+: TWO HANDS (first hand = tens digit, second hand = units digit)
   - Example: #13 = one hand shows 1 (index finger), other hand shows 3 (three fingers)
   - Closed fist = 0 (e.g., #10 = 1 finger + closed fist)
4. FOUL TYPE gestures:
   - Hands on both hips = BLOCKING foul
   - Fist punch into open palm = CHARGING foul
   - One hand chops the other wrist = HAND CHECK foul
   - Grab wrist = HOLDING foul
   - Palm push forward = PUSHING foul

ALSO LOOK FOR:
- On-screen text overlays: "FOUL ON #X", "PERSONAL FOUL", player name graphic
- Player on the floor or holding their body (fouled player)
- Free throw setup (players lining up on lane = foul was called)

Return ONLY valid JSON:
{{
  "foul_called": true or false,
  "referee_visible": true or false,
  "jersey_number": null or string (e.g. "7", "13"),
  "player_name": null or string (exact name from roster if number matches),
  "team": "titans" or "rival" or null,
  "foul_type": null or "blocking" or "charging" or "holding" or "hand_check" or "pushing" or "unsportsmanlike" or "personal",
  "confidence": 0.0 to 1.0,
  "reasoning": "what specific visual evidence supports this (e.g. referee holds up 3 fingers = #3)"
}}

Confidence guide:
- 0.9+ : referee clearly visible, finger count readable, jersey number confirmed
- 0.7-0.9 : referee visible but number inference from context
- 0.5-0.7 : screen text overlay or player reaction visible
- below 0.5 : uncertain"""

    try:
        content = []
        for path in frame_paths:
            compressed = _compress_frame(path, max_dim=640)
            img_data = base64.b64encode(compressed).decode()
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_data}})
        content.append({"type": "text", "text": prompt})

        def _call():
            return get_client().messages.create(
                model="claude-sonnet-4-6",
                max_tokens=400,
                messages=[{"role": "user", "content": content}]
            ).content[0].text

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, _call)
        raw = re.sub(r"```(?:json)?\s*", "", raw.strip()).strip()
        s, e = raw.find('{'), raw.rfind('}')
        if s != -1 and e != -1:
            result = json.loads(raw[s:e+1])
            # Auto-fill player name from jersey_map if not already set
            if result.get("jersey_number") and not result.get("player_name"):
                result["player_name"] = jersey_map.get(str(result["jersey_number"]))
            return result
    except Exception as err:
        print(f"referee analysis error: {err}")
    return {"foul_called": False}


# ── Timeout / stoppage stats screen reading ──────────────────────────────────

async def analyze_timeout_screen(frame_path: str, players: list[str]) -> dict:
    """
    During a timeout or halftime, the broadcast often shows individual player stats.
    Read the stats overlay if visible.

    Returns {"has_stats": bool, "player_stats": {player_name: {stat_key: value}},
             "quarter": str, "is_halftime": bool}
    """
    try:
        compressed = _compress_frame(frame_path, max_dim=720)
        img_data = base64.b64encode(compressed).decode()

        roster_str = ", ".join(players)
        prompt = f"""Basketball broadcast frame during a STOPPAGE (timeout or halftime).

Titans roster: {roster_str}

TASK: Look for a STATISTICS OVERLAY on screen — many broadcasts show individual player stats during timeouts.

What to look for:
- A table or list with player names and numbers (points, rebounds, assists, fouls, etc.)
- Halftime stats summary with individual leaders
- Statistical leader graphics: "POINTS LEADER: GABAY 12"
- Team stats overlay with individual player rows

Return ONLY valid JSON:
{{
  "has_stats": true or false,
  "is_timeout": true or false,
  "is_halftime": true or false,
  "player_stats": {{
    "player_name": {{
      "PTS": null or number,
      "REB": null or number,
      "AST": null or number,
      "STL": null or number,
      "BLK": null or number,
      "FOUL": null or number
    }}
  }},
  "text_visible": "describe any text or graphics you see on screen"
}}

Only include players where stats are clearly visible. If no stats overlay, return has_stats: false."""

        def _call():
            return get_client().messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=400,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_data}},
                    {"type": "text", "text": prompt},
                ]}]
            ).content[0].text

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, _call)
        raw = re.sub(r"```(?:json)?\s*", "", raw.strip()).strip()
        s, e = raw.find('{'), raw.rfind('}')
        if s != -1 and e != -1:
            return json.loads(raw[s:e+1])
    except Exception as err:
        print(f"timeout screen error: {err}")
    return {"has_stats": False}
