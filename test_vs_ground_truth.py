"""
Test: Full Auto pipeline vs ground-truth manual stats
Game: Titans vs La Salle (CLS) 65-62 — 20 Mayo 2026
URL: https://www.youtube.com/watch?v=Js6ReUCdhAE

HOW IT WORKS:
  1. Open WebSocket to receive events in real-time
  2. Trigger pipeline via HTTP POST to /api/full-auto
  3. Collect all ai_event messages and compare vs ground truth
"""

import asyncio, json, time, urllib.request
import websockets

URL = "https://www.youtube.com/watch?v=Js6ReUCdhAE"
SID = "test_cls_v2"
BASE = "http://localhost:8000"

PLAYERS = [
    "Aaron Breziner", "Andre Setton", "Joseph Gabay",
    "Daniel Abadi", "Ilay Mendelson", "Alberto Yahni",
    "Zury Attia", "Saul Piciotto", "Ramon Malca",
    "Ariel Gean", "Ariel Ghershfeld", "Toby Burstein",
]

GROUND_TRUTH = {
    "Aaron Breziner": {"2PT_MADE":3,"2PT_ATT":12,"3PT_MADE":0,"3PT_ATT":1,"FT_MADE":2,"FT_ATT":3,"TOV":0,"FOUL":4,"REB_OFF":3,"REB_DEF":8,"AST":4},
    "Andre Setton":   {"2PT_MADE":11,"2PT_ATT":17,"3PT_MADE":0,"3PT_ATT":1,"FT_MADE":5,"FT_ATT":12,"TOV":7,"FOUL":4,"REB_OFF":4,"REB_DEF":8,"AST":0},
    "Joseph Gabay":   {"2PT_MADE":5,"2PT_ATT":8,"3PT_MADE":0,"3PT_ATT":1,"FT_MADE":1,"FT_ATT":1,"TOV":4,"FOUL":5,"REB_OFF":2,"REB_DEF":5,"AST":0},
    "Daniel Abadi":   {"2PT_MADE":4,"2PT_ATT":12,"3PT_MADE":0,"3PT_ATT":0,"FT_MADE":1,"FT_ATT":3,"TOV":4,"FOUL":2,"REB_OFF":4,"REB_DEF":8,"AST":1},
    "Ilay Mendelson": {"2PT_MADE":4,"2PT_ATT":7,"3PT_MADE":0,"3PT_ATT":1,"FT_MADE":2,"FT_ATT":4,"TOV":4,"FOUL":3,"REB_OFF":3,"REB_DEF":7,"AST":0},
    "Alberto Yahni":  {"2PT_MADE":0,"2PT_ATT":2,"TOV":2,"FOUL":0,"REB_OFF":0,"REB_DEF":1,"AST":2},
}

PAYLOAD = {
    "url": URL,
    "session_id": SID,
    "players": PLAYERS,
    "jersey_map": {"_titans_color": "gray/white"},
    "score_interval": 3,
    "player_profiles": {
        "Andre Setton":   "tallest player on team, center/power forward, strong physical build, dominant in the paint, main scorer with 27 points this game",
        "Aaron Breziner": "point guard, handles the ball most of the time, medium height, distributor and facilitator, 4 assists this game",
        "Joseph Gabay":   "quick combo guard, drives aggressively to the basket, medium-short height, fouled out (5 fouls) this game",
        "Daniel Abadi":   "power forward, strong rebounder (12 rebounds this game), inside scorer, similar build to Setton but slightly shorter",
        "Ilay Mendelson": "shooting guard, mid-range and inside scorer, medium height and build",
        "Alberto Yahni":  "role player, limited minutes, shorter build",
    },
    "titans_jersey_color": "gray/white",   # Titans ALWAYS wear gray/white — do not override
    "rival_jersey_color": "light blue",
}

def pts(s):
    return s.get("2PT_MADE",0)*2 + s.get("3PT_MADE",0)*3 + s.get("FT_MADE",0)

def http_post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{BASE}{path}", data=data,
                                  headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())

async def run():
    print(f"\n{'='*60}")
    print("TITANS AUTO TRACKER — TEST vs GROUND TRUTH")
    print(f"Game: Titans vs La Salle (CLS) 65-62")
    print(f"{'='*60}\n")

    ai_stats = {p: {k:0 for k in ["2PT_MADE","2PT_ATT","3PT_MADE","3PT_ATT",
                                    "FT_MADE","FT_ATT","REB_OFF","REB_DEF",
                                    "AST","TOV","FOUL"]} for p in PLAYERS}
    events_log = []
    start_time = time.time()
    last_progress_print = time.time()

    async with websockets.connect(
        f"ws://localhost:8000/ws/{SID}",
        max_size=10*1024*1024,
        ping_interval=20,
        ping_timeout=60,
        open_timeout=15,
    ) as ws:
        # Trigger pipeline via HTTP POST (not WebSocket message)
        resp = http_post("/api/full-auto", PAYLOAD)
        print(f"Pipeline triggered: {resp}\n")

        done = False
        while not done:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=120)
                msg = json.loads(raw)
                mtype = msg.get("type","")

                if mtype == "status":
                    lvl = msg.get("level","info")
                    txt = msg.get("msg","")
                    icon = {"success":"✅","warn":"⚠️","error":"❌"}.get(lvl,"ℹ️")
                    print(f"{icon} {txt}", flush=True)

                elif mtype == "ai_event":
                    player = msg.get("player","")
                    stat   = msg.get("stat","")
                    conf   = msg.get("confidence",0)
                    ts_vid = msg.get("video_ts","")
                    src    = msg.get("source","")
                    events_log.append(msg)

                    if player in ai_stats and conf >= 0.55:
                        s = ai_stats[player]
                        if stat == "2PT_MADE":   s["2PT_MADE"]+=1; s["2PT_ATT"]+=1
                        elif stat == "3PT_MADE": s["3PT_MADE"]+=1; s["3PT_ATT"]+=1
                        elif stat == "FT_MADE":  s["FT_MADE"]+=1;  s["FT_ATT"]+=1
                        elif stat == "2PT_MISS": s["2PT_ATT"]+=1
                        elif stat == "3PT_MISS": s["3PT_ATT"]+=1
                        elif stat == "FT_MISS":  s["FT_ATT"]+=1
                        elif stat in s:          s[stat]+=1

                    gt_pts_str = f"(GT:{pts(GROUND_TRUTH[player])})" if player in GROUND_TRUTH else ""
                    print(f"  🏀 [{src}] {player}: {stat} @{ts_vid} [{conf:.0%}] ai_pts={pts(ai_stats.get(player,{}))} {gt_pts_str}", flush=True)

                elif mtype == "auto_progress":
                    pct = msg.get("pct",0)
                    elapsed = int(time.time() - start_time)
                    if time.time() - last_progress_print > 60:
                        print(f"  📊 {pct}% complete ({elapsed//60}m{elapsed%60}s elapsed)", flush=True)
                        last_progress_print = time.time()

                elif mtype == "jersey_update":
                    jmap = msg.get("map",{})
                    if jmap:
                        print(f"  🔢 Jerseys: {jmap}", flush=True)

                elif mtype == "substitution":
                    print(f"  🔄 Sub @{msg.get('video_ts','')}: {msg.get('sub_out','?')} → {msg.get('sub_in','?')}", flush=True)

                elif mtype == "auto_done":
                    print("\n✅ Pipeline DONE!", flush=True)
                    done = True

                elif mtype == "error":
                    print(f"❌ Error: {msg}", flush=True)
                    done = True

            except asyncio.TimeoutError:
                elapsed = int(time.time() - start_time)
                print(f"  ⏳ {elapsed//60}m{elapsed%60}s — still running...", flush=True)

    # ── Comparison report ─────────────────────────────────────────────────
    elapsed_total = int(time.time() - start_time)
    print(f"\n{'='*60}")
    print(f"COMPARISON: AI vs Manual — {elapsed_total//60}m{elapsed_total%60}s total")
    print(f"{'='*60}")
    print(f"{'Jugador':<20} {'AI':>5} {'GT':>5}  {'2PT':>5} {'3PT':>4} {'TL':>4} {'FALT':>5} {'TOV':>4} {'REB.D':>6} {'AST*':>5}")
    print("-"*70)

    match_count = 0
    for p in PLAYERS:
        gt = GROUND_TRUTH.get(p)
        ai = ai_stats[p]
        ai_p = pts(ai)
        if gt is None and ai_p == 0:
            continue
        gt_p = pts(gt) if gt else 0
        diff = abs(ai_p - gt_p)
        flag = "✓" if diff <= 3 else "✗"
        if diff <= 3 and gt: match_count += 1
        print(f"{p:<20} {ai_p:>5} {gt_p:>5}  "
              f"{ai['2PT_MADE']:>2}/{ai.get('2PT_ATT',0):<2} "
              f"{ai['3PT_MADE']:>3} "
              f"{ai['FT_MADE']:>2}/{ai.get('FT_ATT',0):<2} "
              f"{ai['FOUL']:>4} "
              f"{ai['TOV']:>4} "
              f"{ai['REB_DEF']:>5} "
              f"{ai['AST']:>5}  {flag}", flush=True)

    gt_count = sum(1 for p in PLAYERS if p in GROUND_TRUTH)
    print(f"\nPuntos dentro de ±3: {match_count}/{gt_count} jugadores")
    print(f"Eventos AI totales: {len(events_log)}")
    print("* AST: precisión AI ~65% — verificar manualmente")

    with open(f"test_results_{SID}.json","w") as f:
        json.dump({"events": events_log, "ai_stats": ai_stats,
                   "ground_truth": GROUND_TRUTH, "elapsed_sec": elapsed_total}, f, indent=2)
    print(f"\nResultados guardados en: test_results_{SID}.json")

asyncio.run(run())
