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
    jersey_map: dict,
    score_before: dict,
    score_after: dict,
    video_ts: str,
    player_profiles: dict | None = None,
    titans_jersey_color: str = "gray/white",
    rival_jersey_color: str = "colored (yellow, red, blue, etc.)",
) -> tuple:
    """
    Copa Talento-optimized multi-signal play analysis.
    Uses jersey colors, crop regions, and physical profiles for player ID.
    Returns (events, jersey_numbers_seen, play_description).
    """
    if not frame_paths:
        return [], {}, ""

    titans_delta = (score_after.get("titans") or 0) - (score_before.get("titans") or 0)
    rival_delta  = (score_after.get("rival") or 0)  - (score_before.get("rival") or 0)

    # Build roster string with jersey numbers and profiles
    roster_lines = []
    for p in players:
        num = next((n for n, name in jersey_map.items() if name == p), "?")
        profile = (player_profiles or {}).get(p, "")
        line = f"  #{num} — {p}"
        if profile:
            line += f" ({profile})"
        roster_lines.append(line)
    roster_str = "\n".join(roster_lines) if roster_lines else "\n".join(f"  {p}" for p in players)

    score_desc = []
    if titans_delta > 0:
        pt_type = {1: "FREE THROW(S)", 2: "2-POINTER", 3: "3-POINTER"}.get(titans_delta, f"+{titans_delta} pts")
        score_desc.append(f"TITANS scored {pt_type} → now {score_after['titans']}")
    if rival_delta > 0:
        pt_type = {1: "FREE THROW(S)", 2: "2-POINTER", 3: "3-POINTER"}.get(rival_delta, f"+{rival_delta} pts")
        score_desc.append(f"RIVAL scored {pt_type} → now {score_after['rival']}")
    if not score_desc:
        score_desc = ["Score unchanged — possible foul, timeout, or dead ball"]
    score_summary = "\n".join(score_desc)

    # Select crop regions based on play type (what to zoom in on)
    crop_instructions = ""
    crop_regions = []
    if titans_delta == 1 or rival_delta == 1:
        # Free throw: shooter alone at foul line, center of court
        crop_regions = [(0.3, 0.35, 0.7, 0.75)]   # center vertically (foul line zone)
        crop_instructions = "IMPORTANT: This is a FREE THROW (+1 pt). The shooter is ALONE at the foul line in the CENTER of the court. The camera often ZOOMS IN for free throws. Look at the zoomed crop image."
    elif titans_delta == 3 or rival_delta == 3:
        # 3-pointer: shooter behind the arc (outer perimeter)
        crop_regions = [(0.0, 0.2, 0.5, 0.65), (0.5, 0.2, 1.0, 0.65)]  # left/right perimeter
        crop_instructions = "IMPORTANT: This is a 3-POINTER. The shooter was BEHIND the 3-point arc (outer edges of the court). Look for a player celebrating or holding their follow-through pose."
    else:
        # 2-pointer: player was near the basket (in the paint)
        crop_regions = [(0.25, 0.3, 0.75, 0.85)]  # paint area
        crop_instructions = "IMPORTANT: This is a 2-POINTER. The scorer was near the BASKET (in the key/paint). Look for the player closest to the basket."

    prompt = f"""You are analyzing frames from a Copa Talento Colegial U18 basketball broadcast in Panama.

═══ BROADCAST PROFILE ═══
• Camera: Fixed elevated wide-angle, full-court view from scorer's table side
• Scoreboard: Bottom-RIGHT corner — team name + score + clock + quarter ("1st", "2nd", etc.)
• TITANS jersey color: {titans_jersey_color} (light gray or white)
• RIVAL jersey color: {rival_jersey_color}
• Referee: Black shirt. Scorer's table crew: white shirts.

═══ WHAT HAPPENED ═══
{score_summary}
Timestamp: {video_ts}
{crop_instructions}

═══ TITANS ROSTER ═══
{roster_str}

═══ YOUR TASK ═══
Step 1 — TEAM CONFIRMATION: Confirm WHICH team scored based on jersey color near the basket.
Step 2 — PLAYER IDENTIFICATION: Use every available signal:
  a) Jersey NUMBER on shirt (front or back) — even partially visible numbers count
  b) Physical appearance: height, build, skin tone, hair
  c) Court position: Who is at the foul line (FT)? Who is in the paint (2PT)? Who is at the arc (3PT)?
  d) Post-play behavior: The scorer often pumps fist, raises arms, looks at bench
  e) Team huddle: After play, teammates often rush to the scorer
  f) Any TEXT OVERLAYS: Player name graphics, "FOUL ON #X", scoreboard text
  g) Player profiles if provided (height, position tendency, etc.)

Step 3 — SECONDARY EVENTS: Also look for:
  • Rebounds after misses (who catches the ball?)
  • Assists (who passed to the scorer right before?)
  • Steals or blocks (if clearly visible)
  • Team fouls on the scoreboard (did it increment?)

Return ONLY valid JSON:
{{
  "events": [
    {{
      "player": "exact name from roster, or UNKNOWN_TITANS, or RIVAL",
      "team": "titans" or "rival",
      "stat": "2PT_MADE|3PT_MADE|FT_MADE|2PT_MISS|3PT_MISS|FT_MISS|REB_OFF|REB_DEF|AST|STL|BLK|FOUL",
      "confidence": 0.0 to 1.0,
      "reasoning": "specific visual evidence: jersey color + number + position + behavior"
    }}
  ],
  "jersey_numbers_seen": {{}},
  "titans_jersey_visible": true or false,
  "play_description": "one sentence describing the play"
}}

Confidence:
• 0.9+: jersey # clearly visible + matches a known play
• 0.75-0.9: jersey partially readable OR physical match + position match
• 0.6-0.75: team confirmed (jersey color) + strong positional inference
• 0.5-0.6: team confirmed but player uncertain → use UNKNOWN_TITANS
• <0.5: do not include"""

    try:
        # Build content: full frames first, then crop zooms, then prompt
        content = []
        for path in frame_paths:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg",
                           "data": _encode_frame(path, max_dim=720)}
            })

        # Add crop zoom images (labeled via text)
        if crop_regions and frame_paths:
            # Use the middle frame for cropping (most likely to show the play)
            mid_frame = frame_paths[len(frame_paths) // 2]
            content.append({"type": "text", "text": "=== ZOOMED CROP OF KEY AREA ==="})
            for region in crop_regions:
                cropped = _crop_and_encode(mid_frame, region, zoom=2.5)
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": cropped}
                })

        content.append({"type": "text", "text": prompt})

        def _call():
            return get_client().messages.create(
                model="claude-sonnet-4-6",
                max_tokens=900,
                messages=[{"role": "user", "content": content}]
            ).content[0].text

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, _call)
        raw = re.sub(r"```(?:json)?\s*", "", raw.strip()).strip()
        s, e = raw.find('{'), raw.rfind('}')
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


def _encode_frame(path: str, max_dim: int = 720, quality: int = 75) -> str:
    """Encode frame as base64 JPEG string for Claude API."""
    return base64.b64encode(_compress_frame(path, max_dim)).decode()


def _crop_and_encode(frame_path: str, region: tuple, zoom: float = 2.0, quality: int = 80) -> str:
    """
    Crop a rectangular region from a frame, zoom it up, and return base64 JPEG.
    region = (left_frac, top_frac, right_frac, bottom_frac) — fractions of image size.
    """
    try:
        img = Image.open(frame_path).convert("RGB")
        w, h = img.size
        l = int(region[0] * w); t = int(region[1] * h)
        r = int(region[2] * w); b = int(region[3] * h)
        crop = img.crop((l, t, r, b))
        new_w = int((r - l) * zoom)
        new_h = int((b - t) * zoom)
        zoomed = crop.resize((max(new_w, 120), max(new_h, 120)), Image.LANCZOS)
        buf = io.BytesIO()
        zoomed.save(buf, "JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"crop error: {e}")
        return _encode_frame(frame_path, max_dim=480)


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


async def scan_frame_for_jerseys(frame_path: str, players: list[str], jersey_map: dict) -> dict:
    """
    Dedicated jersey-number extraction scan — optimized for close-up or warmup frames
    where players are larger and numbers more readable.
    Returns {"jersey_map_updates": {"7": "Gabay"}, "descriptions": {"7": "tall player, light skin"}}
    """
    roster_str = "\n".join(f"  - {p}" for p in players)
    already_known = "\n".join(f"  #{k} = {v}" for k, v in jersey_map.items()) if jersey_map else "  (none yet)"

    prompt = f"""Scan this basketball frame for jersey numbers. This may be a warmup, timeout, or sideline shot where players are CLOSE to the camera.

Titans roster (find jersey numbers for as many as possible):
{roster_str}

Already known jersey → player mappings:
{already_known}

INSTRUCTIONS:
1. Look at EVERY visible jersey number in the frame
2. For Titans players (gray/white jerseys): read the number, match to roster name if possible
3. A player's number is on their CHEST and BACK
4. Even if partially obscured, write down what you can read
5. Note physical description to help match to roster (height, build, hair, skin)

Return ONLY valid JSON:
{{
  "jerseys_found": [
    {{
      "number": "7",
      "team": "titans" or "rival",
      "confidence": 0.0 to 1.0,
      "player_name": "exact roster name or null",
      "description": "tall, dark hair, light skin — standing near scorer table"
    }}
  ],
  "close_up": true or false,
  "warmup_or_timeout": true or false
}}"""

    try:
        img_data = _encode_frame(frame_path, max_dim=720, quality=85)
        # Also add zoomed crops of left and right sides (where players often stand)
        left_crop = _crop_and_encode(frame_path, (0.0, 0.1, 0.5, 0.9), zoom=2.0)
        right_crop = _crop_and_encode(frame_path, (0.5, 0.1, 1.0, 0.9), zoom=2.0)

        def _call():
            return get_client().messages.create(
                model="claude-sonnet-4-6",
                max_tokens=600,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_data}},
                    {"type": "text", "text": "Full frame:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": left_crop}},
                    {"type": "text", "text": "Zoomed left side:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": right_crop}},
                    {"type": "text", "text": "Zoomed right side:"},
                    {"type": "text", "text": prompt},
                ]}]
            ).content[0].text

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, _call)
        raw = re.sub(r"```(?:json)?\s*", "", raw.strip()).strip()
        s, e = raw.find('{'), raw.rfind('}')
        if s != -1 and e != -1:
            data = json.loads(raw[s:e+1])
            result = {"jersey_map_updates": {}, "descriptions": {}}
            for j in data.get("jerseys_found", []):
                num = str(j.get("number", "")).strip()
                if num and j.get("team") == "titans" and j.get("confidence", 0) >= 0.55:
                    pname = j.get("player_name")
                    if pname and pname in players:
                        result["jersey_map_updates"][num] = pname
                    if j.get("description"):
                        result["descriptions"][num] = j["description"]
            return result
    except Exception as err:
        print(f"jersey scan error: {err}")
    return {"jersey_map_updates": {}, "descriptions": {}}


async def detect_team_jersey_colors(frame_path: str) -> dict:
    """
    Detect the jersey colors of each team from a frame.
    Returns {"titans_color": "gray/white", "rival_color": "yellow"}
    Used once at the start to customize prompts.
    """
    try:
        img_data = _encode_frame(frame_path, max_dim=480)
        prompt = """Basketball broadcast frame. Two teams are playing.
One team is called TITANS (look for this name on the scoreboard).

Identify the jersey colors:
Return ONLY JSON: {"titans_color": "color description", "rival_color": "color description", "rival_name": "team name from scoreboard or null"}
Example: {"titans_color": "light gray/white", "rival_color": "bright yellow", "rival_name": "Aguilas"}"""

        def _call():
            return get_client().messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=80,
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
        print(f"color detect error: {err}")
    return {"titans_color": "gray/white", "rival_color": "colored"}


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
