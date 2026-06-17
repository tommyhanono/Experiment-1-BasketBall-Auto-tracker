import os, json, re, base64, asyncio
import anthropic

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
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text

    loop = asyncio.get_event_loop()
    try:
        raw = await loop.run_in_executor(None, _call)
        raw = raw.strip()
        # Extract JSON if wrapped in code blocks
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
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
        prompt = f"""Analyze this basketball broadcast frame.

Titans players: {roster_str}
Context: {context}

Extract from the image:
1. Score (Titans score and opponent score if visible)
2. Game clock / quarter
3. Any player name graphics or stat overlays visible on screen
4. Any play-by-play text graphics

Return ONLY valid JSON:
{{
  "titans_score": null or number,
  "rival_score": null or number,
  "quarter": null or string,
  "clock": null or string,
  "player_events": [],
  "text_on_screen": "any text visible"
}}"""

        def _call():
            resp = get_client().messages.create(
                model="claude-sonnet-4-6",
                max_tokens=400,
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
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        print(f"Frame analyzer error: {e}")
    return {}
