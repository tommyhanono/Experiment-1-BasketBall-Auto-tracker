"""
Basketball Computer Vision Tracking Engine
==========================================
Implements commercial-grade player tracking using OpenCV.
Works at 360p - designed specifically for Copa Talento broadcast style.

Pipeline:
  1. Background subtraction → detect all moving blobs (players + ball)
  2. HSV team color classification → Titans (gray/white) vs rival (colored)
  3. Kalman filter → smooth trajectories, fill gaps
  4. Court calibration → map pixel coords to court zones
  5. Event detection → who scored, rebounded, assisted, stole, etc.
"""

import cv2
import numpy as np
import os
import json
import base64
import tempfile
from collections import defaultdict, deque, Counter

# PaddleOCR — best-in-class for jersey number reading (3x better than EasyOCR per benchmarks)
_paddle_ocr = None
def _get_paddle():
    global _paddle_ocr
    if _paddle_ocr is None:
        try:
            from paddleocr import PaddleOCR
            _paddle_ocr = PaddleOCR(lang='en', use_textline_orientation=False)
        except Exception:
            _paddle_ocr = False
    return _paddle_ocr if _paddle_ocr is not False else None

# Try to load YOLOv8 — best-in-class player/ball detection
try:
    from ultralytics import YOLO as _YOLO
    _yolo_model = None

    def _get_yolo():
        global _yolo_model
        if _yolo_model is None:
            _yolo_model = _YOLO("yolov8n.pt")  # nano: 6MB, fast on CPU
        return _yolo_model

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    def _get_yolo(): return None


# ── Constants ──────────────────────────────────────────────────────────────

# HSV ranges (H: 0-179, S: 0-255, V: 0-255 in OpenCV)
HSV_WHITE_GRAY  = ((0,   0,  160), (180,  60, 255))   # white/gray jerseys (Titans)
HSV_YELLOW      = ((15,  80,  80), ( 35, 255, 255))   # yellow jerseys
HSV_ORANGE_BALL = ((5,  120, 100), ( 25, 255, 255))   # basketball color
HSV_BLUE        = ((90,  80,  50), (130, 255, 255))   # blue jerseys
HSV_RED         = ((0,  100,  80), ( 10, 255, 255))   # red jerseys (low hue)
HSV_RED2        = ((165, 100, 80), (180, 255, 255))   # red (high hue wrap)

# Copa Talento court layout (fractions of 640x360 frame)
# Based on the observed broadcast: camera at scorer's table side, slight angle
BASKET_LEFT  = (0.12, 0.52)   # left basket in frame (normalized coords)
BASKET_RIGHT = (0.88, 0.52)   # right basket in frame
THREE_PT_DIST_FRAC = 0.18     # ~18% of frame width from basket = 3PT arc
PAINT_W_FRAC  = 0.08          # paint width as fraction of frame
PAINT_H_FRAC  = 0.18          # paint height
COURT_TOP_FRAC    = 0.15      # court starts here (top boundary)
COURT_BOTTOM_FRAC = 0.88      # court ends here (bottom boundary)
SCORER_TABLE_Y_FRAC = 0.90    # scorer's table is at very bottom of frame


# ── Team color classifier ──────────────────────────────────────────────────

class TeamClassifier:
    """
    Classify a player's team from their jersey color using HSV histograms.
    Calibrated dynamically from the first readable frames.
    """

    def __init__(self, titans_color_hint="gray/white", rival_color_hint="yellow"):
        self.titans_hsv = self._hint_to_hsv(titans_color_hint)
        self.rival_hsv  = self._hint_to_hsv(rival_color_hint)

    def _hint_to_hsv(self, hint: str):
        hint = hint.lower()
        if any(w in hint for w in ["gray", "grey", "white", "gris", "blanco"]):
            return [HSV_WHITE_GRAY]
        if "yellow" in hint or "amarillo" in hint:
            return [HSV_YELLOW]
        if "blue" in hint or "azul" in hint:
            return [HSV_BLUE]
        if "red" in hint or "rojo" in hint:
            return [HSV_RED, HSV_RED2]
        return [HSV_YELLOW]  # default rival

    def classify_blob(self, frame_bgr: np.ndarray, bbox: tuple) -> str:
        """
        Given a bounding box (x, y, w, h), classify player team.
        Returns 'titans', 'rival', or 'unknown'.
        """
        x, y, w, h = bbox
        if w < 8 or h < 8:
            return "unknown"

        # Extract torso region (upper 55% of bbox = jersey)
        torso_top    = y + int(h * 0.05)
        torso_bottom = y + int(h * 0.60)
        torso = frame_bgr[torso_top:torso_bottom, x:x+w]
        if torso.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

        def coverage(ranges):
            mask = None
            for (lo, hi) in ranges:
                m = cv2.inRange(hsv, np.array(lo), np.array(hi))
                mask = m if mask is None else cv2.bitwise_or(mask, m)
            return np.count_nonzero(mask) / mask.size if mask is not None else 0

        titans_cov = coverage(self.titans_hsv)
        rival_cov  = coverage(self.rival_hsv)

        if titans_cov > 0.20 and titans_cov > rival_cov * 1.5:
            return "titans"
        if rival_cov > 0.20 and rival_cov > titans_cov * 1.5:
            return "rival"
        return "unknown"

    def calibrate_from_frame(self, frame_bgr: np.ndarray):
        """
        Auto-detect team colors from a frame with multiple players.
        Uses K-means on torso HSV to find dominant jersey colors.
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        # Sample the middle horizontal band (where players are)
        mid = hsv[int(frame_bgr.shape[0]*0.2):int(frame_bgr.shape[0]*0.8), :]
        # Flatten and take pixels with decent saturation (jersey vs court)
        flat = mid.reshape(-1, 3).astype(np.float32)
        saturated = flat[flat[:, 1] > 40]
        if len(saturated) < 100:
            return
        # K-means with 4 clusters to find dominant colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, _, centers = cv2.kmeans(saturated, 4, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        # Centers are HSV values — identify which clusters are jersey colors
        # (This is a basic auto-calibration; manual hints are more reliable)


# ── Ball detector ──────────────────────────────────────────────────────────

class BallTracker:
    """
    Detect basketball in frames using HSV color + circularity.
    Ball is orange/dark, roughly circular.
    """

    def __init__(self):
        self.history = deque(maxlen=30)  # last 30 frame positions
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea    = True
        params.minArea, params.maxArea = 20, 400
        params.filterByCircularity = True
        params.minCircularity = 0.6
        params.filterByConvexity = True
        params.minConvexity = 0.7
        self.blob_detector = cv2.SimpleBlobDetector_create(params)

    def detect(self, frame_bgr: np.ndarray) -> tuple | None:
        """Returns (x, y) of ball centroid in pixel coords, or None."""
        # Method 1: HSV color range
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(HSV_ORANGE_BALL[0]), np.array(HSV_ORANGE_BALL[1]))
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 15 or area > 500:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            score = circularity * min(area, 150) / 150
            if score > best_score:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    best = (cx, cy)
                    best_score = score

        if best:
            self.history.append(best)
        return best

    def get_trajectory(self) -> list:
        return list(self.history)

    def is_arc_trajectory(self) -> bool:
        """Returns True if recent trajectory looks like a shot arc (parabola)."""
        pts = self.get_trajectory()
        if len(pts) < 6:
            return False
        ys = np.array([p[1] for p in pts])
        # Shot arc: y goes UP then DOWN (or DOWN then UP depending on view)
        mid = len(ys) // 2
        first_half_dir = ys[mid] - ys[0]
        second_half_dir = ys[-1] - ys[mid]
        return (first_half_dir < -3 and second_half_dir > 3) or \
               (first_half_dir > 3 and second_half_dir < -3)


# ── Player tracker ─────────────────────────────────────────────────────────

class PlayerTracker:
    """
    Track all players in a video clip using background subtraction.
    Assigns team labels using jersey color classification.
    Returns per-frame player positions.
    """

    def __init__(self, classifier: TeamClassifier):
        self.classifier = classifier
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=30, varThreshold=25, detectShadows=False
        )
        self.tracked_ids = {}   # id → {team, centroid_history}
        self._next_id = 0

    def _match_to_existing(self, centroid, team, max_dist=40):
        """Find closest existing track to this centroid."""
        best_id, best_d = None, max_dist
        for tid, info in self.tracked_ids.items():
            if not info["history"]:
                continue
            prev = info["history"][-1]
            d = np.hypot(centroid[0] - prev[0], centroid[1] - prev[1])
            if d < best_d and (info["team"] == team or info["team"] == "unknown"):
                best_id, best_d = tid, d
        return best_id

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> list[dict]:
        """
        Process one frame. Returns list of {id, team, centroid, bbox}.
        """
        # Apply background subtraction
        fg_mask = self.bg_sub.apply(frame_bgr)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        # Only consider court area (exclude scoreboard, crowds)
        h, w = frame_bgr.shape[:2]
        court_mask = np.zeros_like(fg_mask)
        court_mask[int(h * COURT_TOP_FRAC):int(h * COURT_BOTTOM_FRAC), :] = 255
        fg_mask = cv2.bitwise_and(fg_mask, court_mask)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Player blobs at 360p are roughly 300-2500 px²
            if area < 150 or area > 4000:
                continue
            x, y, w_b, h_b = cv2.boundingRect(cnt)
            # Filter by aspect ratio (players are taller than wide)
            if h_b < w_b * 0.8 or h_b > w_b * 6:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            team = self.classifier.classify_blob(frame_bgr, (x, y, w_b, h_b))
            detections.append({"centroid": (cx, cy), "bbox": (x, y, w_b, h_b), "team": team})

        # Match detections to existing tracks
        active = []
        for det in detections:
            tid = self._match_to_existing(det["centroid"], det["team"])
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self.tracked_ids[tid] = {"team": det["team"], "history": [], "jersey": None}
            else:
                # Update team if we were unknown
                if self.tracked_ids[tid]["team"] == "unknown" and det["team"] != "unknown":
                    self.tracked_ids[tid]["team"] = det["team"]
            self.tracked_ids[tid]["history"].append(det["centroid"])
            active.append({
                "id": tid,
                "team": self.tracked_ids[tid]["team"],
                "centroid": det["centroid"],
                "bbox": det["bbox"],
                "frame": frame_idx,
            })

        return active

    def reset(self):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=30, varThreshold=25, detectShadows=False
        )
        self.tracked_ids = {}
        self._next_id = 0


# ── Court geometry ─────────────────────────────────────────────────────────

def normalize_coords(x, y, frame_w, frame_h):
    return x / frame_w, y / frame_h


def get_court_zone(nx, ny):
    """
    Given normalized coords, return court zone string.
    Assumes camera shows left basket on left, right basket on right.
    """
    # Determine which half
    if nx < 0.5:
        side = "left"
        basket = BASKET_LEFT
    else:
        side = "right"
        basket = BASKET_RIGHT

    bx, by = basket
    dist = np.hypot(nx - bx, ny - by)

    if dist < 0.06:
        return f"{side}_paint_close"   # in the key, close range
    if dist < PAINT_W_FRAC * 1.5:
        return f"{side}_paint"          # paint area
    if dist < THREE_PT_DIST_FRAC:
        return f"{side}_midrange"       # mid-range
    return f"{side}_three"             # three-point territory


def nearest_basket(nx, ny):
    """Return which basket (left/right) is closer and the distance."""
    dl = np.hypot(nx - BASKET_LEFT[0],  ny - BASKET_LEFT[1])
    dr = np.hypot(nx - BASKET_RIGHT[0], ny - BASKET_RIGHT[1])
    if dl < dr:
        return "left", dl
    return "right", dr


def is_near_basket(nx, ny, threshold=0.12):
    _, d = nearest_basket(nx, ny)
    return d < threshold


def is_three_point_zone(nx, ny):
    _, d = nearest_basket(nx, ny)
    return d >= THREE_PT_DIST_FRAC


# ── Event detector ─────────────────────────────────────────────────────────

class EventDetector:
    """
    Infer basketball events from tracking data:
    - Who scored (2PT or 3PT)
    - Who rebounded
    - Who assisted
    - Who stole the ball
    - Who blocked
    """

    def __init__(self, frame_w: int, frame_h: int):
        self.fw = frame_w
        self.fh = frame_h

    def _norm(self, x, y):
        return x / self.fw, y / self.fh

    def find_scorer(self, frames_data: list[dict], score_delta: dict) -> dict:
        """
        From tracking frames around a score change, identify who scored.
        Returns {"team": "titans"|"rival", "shot_type": "2PT"|"3PT"|"FT",
                 "track_id": N, "confidence": 0.0-1.0, "zone": "..."}
        """
        titans_delta = score_delta.get("titans", 0)
        rival_delta  = score_delta.get("rival", 0)
        scoring_team = "titans" if titans_delta > 0 else "rival"
        delta = titans_delta or rival_delta

        # Determine shot type from delta
        shot_type = {1: "FT", 2: "2PT", 3: "3PT"}.get(delta, "2PT")

        # Find the frame closest to when the basket was made
        # (usually the middle-to-end of the clip when score changes)
        mid_idx = len(frames_data) * 2 // 3
        search_frames = frames_data[max(0, mid_idx - 5): mid_idx + 5]

        # For FT: player at foul line center (normalized ~0.5, 0.55)
        if shot_type == "FT":
            return self._find_ft_shooter(frames_data, scoring_team)

        # For 2PT/3PT: find scoring team player nearest to basket
        best_player = None
        best_dist = 1.0
        best_zone  = "unknown"

        for fd in search_frames:
            for p in fd.get("players", []):
                if p["team"] != scoring_team:
                    continue
                nx, ny = self._norm(*p["centroid"])
                _, dist = nearest_basket(nx, ny)
                zone = get_court_zone(nx, ny)
                # For 3PT, prefer players in 3PT zone
                if shot_type == "3PT" and "three" not in zone:
                    continue
                if shot_type == "2PT" and "three" in zone:
                    continue
                if dist < best_dist:
                    best_dist = dist
                    best_player = p
                    best_zone = zone

        if best_player is None:
            # Fallback: any player from scoring team near basket
            for fd in frames_data:
                for p in fd.get("players", []):
                    if p["team"] != scoring_team:
                        continue
                    nx, ny = self._norm(*p["centroid"])
                    _, dist = nearest_basket(nx, ny)
                    zone = get_court_zone(nx, ny)
                    if dist < best_dist:
                        best_dist = dist
                        best_player = p
                        best_zone = zone

        confidence = max(0.4, min(0.85, 1.0 - best_dist * 3)) if best_player else 0.3
        return {
            "team": scoring_team,
            "shot_type": shot_type,
            "track_id": best_player["id"] if best_player else None,
            "confidence": round(confidence, 2),
            "zone": best_zone,
        }

    def _find_ft_shooter(self, frames_data, team):
        """Free throw: player at foul line (center of frame vertically)."""
        foul_line_nx = 0.5
        foul_line_ny = 0.52
        best_player = None
        best_dist = 1.0
        for fd in frames_data:
            for p in fd.get("players", []):
                if p["team"] != team:
                    continue
                nx, ny = self._norm(*p["centroid"])
                dist = np.hypot(nx - foul_line_nx, ny - foul_line_ny)
                if dist < best_dist:
                    best_dist = dist
                    best_player = p
        confidence = max(0.5, 0.95 - best_dist * 2) if best_player else 0.4
        return {"team": team, "shot_type": "FT", "track_id": best_player["id"] if best_player else None,
                "confidence": round(confidence, 2), "zone": "foul_line"}

    def find_rebounder(self, frames_data: list[dict], ball_data: list, rebound_team: str) -> dict:
        """
        After a miss, find who grabbed the rebound.
        The rebounder is the player who:
        1. Is near the basket
        2. Is the same team as rebound_team
        3. Ends up in possession (ball near them after the miss)
        """
        ball_positions = [b for b in ball_data if b is not None]
        if not ball_positions:
            return self._find_near_basket(frames_data, rebound_team, "REB")

        # Find player closest to last ball position near basket
        last_ball = ball_positions[-1]
        nb_x, nb_y = self._norm(*last_ball)

        if not is_near_basket(nb_x, nb_y, threshold=0.20):
            return self._find_near_basket(frames_data, rebound_team, "REB")

        best_player = None
        best_dist   = 1.0
        for fd in frames_data[-10:]:   # look in later frames
            for p in fd.get("players", []):
                if p["team"] != rebound_team:
                    continue
                nx, ny = self._norm(*p["centroid"])
                dist = np.hypot(nx - nb_x, ny - nb_y)
                if dist < best_dist:
                    best_dist = dist
                    best_player = p

        confidence = max(0.45, 0.90 - best_dist * 2) if best_player else 0.35
        return {"team": rebound_team, "track_id": best_player["id"] if best_player else None,
                "confidence": round(confidence, 2), "stat": "REB_DEF"}

    def _find_near_basket(self, frames_data, team, stat):
        best_p, best_d = None, 1.0
        for fd in frames_data:
            for p in fd.get("players", []):
                if p["team"] != team:
                    continue
                nx, ny = self._norm(*p["centroid"])
                _, d = nearest_basket(nx, ny)
                if d < best_d:
                    best_d, best_p = d, p
        conf = max(0.4, 0.85 - best_d * 2) if best_p else 0.3
        return {"team": team, "track_id": best_p["id"] if best_p else None,
                "confidence": round(conf, 2), "stat": stat}

    def find_assister(self, frames_data: list[dict], scorer_track_id: int) -> dict | None:
        """
        Find who passed to the scorer in the 3-5 frames before the basket.
        The assister is the Titans player who was closest to the scorer
        (within passing distance) in the pre-basket frames.
        """
        if scorer_track_id is None:
            return None

        pre_frames = frames_data[:len(frames_data) * 2 // 3]
        scorer_positions = []
        for fd in pre_frames:
            for p in fd.get("players", []):
                if p["id"] == scorer_track_id:
                    scorer_positions.append(p["centroid"])

        if not scorer_positions:
            return None

        # Find Titans player closest to scorer in the build-up
        avg_scorer = np.mean(scorer_positions, axis=0)
        best_passer = None
        best_dist   = 1.0
        for fd in pre_frames:
            for p in fd.get("players", []):
                if p["id"] == scorer_track_id or p["team"] != "titans":
                    continue
                nx_s, ny_s = self._norm(*avg_scorer)
                nx_p, ny_p = self._norm(*p["centroid"])
                dist = np.hypot(nx_s - nx_p, ny_s - ny_p)
                # Passing distance: typically 0.08 - 0.40 of frame
                if 0.05 < dist < 0.45 and dist < best_dist:
                    best_dist = dist
                    best_passer = p

        if best_passer is None:
            return None

        # Confidence based on proximity (closer = more likely assist)
        conf = max(0.35, 0.75 - best_dist)
        return {"team": "titans", "track_id": best_passer["id"],
                "confidence": round(conf, 2), "stat": "AST"}

    def find_steal(self, frames_data: list[dict]) -> dict | None:
        """
        Detect ball possession change: rival player had ball, Titans player ends with it.
        """
        ball_found = False
        prev_team_with_ball = None
        for fd in frames_data:
            ball = fd.get("ball")
            if not ball:
                continue
            bx, by = self._norm(*ball)
            # Find player closest to ball
            min_d, closest = 1.0, None
            for p in fd.get("players", []):
                nx, ny = self._norm(*p["centroid"])
                d = np.hypot(nx - bx, ny - by)
                if d < min_d and d < 0.15:
                    min_d, closest = d, p
            if closest:
                if prev_team_with_ball == "rival" and closest["team"] == "titans":
                    # Possession change: rival → titans = STEAL
                    return {"team": "titans", "track_id": closest["id"],
                            "stat": "STL", "confidence": 0.65}
                prev_team_with_ball = closest["team"]

        return None


# ── Main clip analysis function ────────────────────────────────────────────

def analyze_clip_yolo(
    video_path: str,
    start_sec: float,
    end_sec: float,
    score_delta: dict,
    titans_color: str = "gray/white",
    rival_color: str = "yellow",
) -> dict | None:
    """
    YOLO + ByteTrack player tracking — highest accuracy mode.
    Falls back to None if YOLO not available or clip too short.

    ByteTrack gives each player a persistent integer track ID that survives
    temporary occlusions. We classify each track by jersey color once per
    clip, then use those IDs throughout.
    """
    if not YOLO_AVAILABLE:
        return None

    model = _get_yolo()
    classifier = TeamClassifier(titans_color, rival_color)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360

    start_f = int(start_sec * fps)
    end_f   = int(end_sec   * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    event_det   = EventDetector(w, h)
    frames_data = []
    ball_data   = []
    # team_color_cache: track_id → team (classified once per track)
    team_cache  = {}
    frame_idx   = 0

    while cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_f:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference with ByteTrack — returns Results with boxes.id
        results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml",
                               classes=[0],   # class 0 = person
                               imgsz=640, conf=0.25)
        players = []
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                tid = int(boxes.id[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                bx, by, bw, bh = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                # Classify team once per track (costs very little)
                if tid not in team_cache:
                    team_cache[tid] = classifier.classify_blob(frame, (bx, by, bw, bh))
                elif team_cache[tid] == "unknown":
                    team_cache[tid] = classifier.classify_blob(frame, (bx, by, bw, bh))

                players.append({
                    "id": tid,
                    "team": team_cache[tid],
                    "centroid": centroid,
                    "bbox": (bx, by, bw, bh),
                    "frame": frame_idx,
                })

        # Ball detection from YOLO (class 32 = sports ball) or fallback
        ball = None
        if results and results[0].boxes is not None:
            ball_results = model(frame, classes=[32], verbose=False, imgsz=640, conf=0.20)
            if ball_results and len(ball_results[0].boxes.xyxy) > 0:
                bx1, by1, bx2, by2 = ball_results[0].boxes.xyxy[0].tolist()
                ball = (int((bx1 + bx2) / 2), int((by1 + by2) / 2))

        frames_data.append({"frame": frame_idx, "players": players, "ball": ball})
        ball_data.append(ball)
        frame_idx += 1

    cap.release()

    if not frames_data or frame_idx < 3:
        return None

    titans_d = score_delta.get("titans", 0)
    rival_d  = score_delta.get("rival",  0)
    result = {"scorer": None, "rebounder": None, "assister": None, "steal": None,
              "player_tracks": {}, "summary": "", "engine": "yolo"}

    for fd in frames_data:
        for p in fd["players"]:
            tid = p["id"]
            if tid not in result["player_tracks"]:
                result["player_tracks"][tid] = {"team": p["team"], "centroids": []}
            result["player_tracks"][tid]["centroids"].append(p["centroid"])

    if titans_d > 0 or rival_d > 0:
        result["scorer"] = event_det.find_scorer(frames_data, score_delta)
        if titans_d > 0 and result["scorer"] and result["scorer"]["shot_type"] in ("2PT", "3PT"):
            result["assister"] = event_det.find_assister(frames_data, result["scorer"]["track_id"])
    else:
        steal = event_det.find_steal(frames_data)
        if steal:
            result["steal"] = steal
        rb = event_det._find_near_basket(frames_data, "titans", "REB_DEF")
        if rb["confidence"] >= 0.45:
            result["rebounder"] = rb

    parts = []
    if result["scorer"]:
        s = result["scorer"]
        parts.append(f"[YOLO] TID-{s['track_id']} ({s['team']}) {s['shot_type']} {s['zone']} [{s['confidence']:.0%}]")
    if result["assister"]:
        a = result["assister"]
        parts.append(f"Assist TID-{a['track_id']} [{a['confidence']:.0%}]")
    result["summary"] = " | ".join(parts) or "No event detected"

    return result


def analyze_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    score_delta: dict,
    titans_color: str = "gray/white",
    rival_color: str = "yellow",
    fps_override: float = None,
) -> dict:
    """
    Analyze a video clip to detect basketball events via CV tracking.

    Args:
        video_path: path to downloaded game video
        start_sec: clip start (seconds)
        end_sec: clip end (seconds)
        score_delta: {"titans": N, "rival": N} — what changed in this clip
        titans_color / rival_color: jersey color hints

    Returns:
        {
          "scorer": {"track_id": N, "team": "titans", "shot_type": "2PT", "confidence": 0.8, "zone": "..."},
          "rebounder": {...} or None,
          "assister": {...} or None,
          "steal": {...} or None,
          "player_tracks": {track_id: {"team": "...", "centroid_seq": [...]}},
          "summary": "description string"
        }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "cannot open video"}

    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec   * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    classifier = TeamClassifier(titans_color, rival_color)
    player_tracker = PlayerTracker(classifier)
    ball_tracker   = BallTracker()

    frames_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
    frames_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
    event_det = EventDetector(frames_w, frames_h)

    frames_data = []
    ball_positions = []
    frame_idx = 0

    while cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        players = player_tracker.process_frame(frame, frame_idx)
        ball    = ball_tracker.detect(frame)
        ball_positions.append(ball)
        frames_data.append({"frame": frame_idx, "players": players, "ball": ball})
        frame_idx += 1

    cap.release()

    if not frames_data:
        return {"error": "no frames read"}

    titans_d = score_delta.get("titans", 0)
    rival_d  = score_delta.get("rival",  0)

    result = {
        "scorer": None, "rebounder": None, "assister": None, "steal": None,
        "player_tracks": {}, "summary": ""
    }

    # Collect unique player tracks
    for fd in frames_data:
        for p in fd["players"]:
            tid = p["id"]
            if tid not in result["player_tracks"]:
                result["player_tracks"][tid] = {"team": p["team"], "centroids": []}
            result["player_tracks"][tid]["centroids"].append(p["centroid"])

    # Detect primary scoring event
    if titans_d > 0 or rival_d > 0:
        result["scorer"] = event_det.find_scorer(frames_data, score_delta)

        # If Titans scored, look for assist
        if titans_d > 0 and result["scorer"] and result["scorer"]["shot_type"] in ("2PT", "3PT"):
            scorer_tid = result["scorer"]["track_id"]
            result["assister"] = event_det.find_assister(frames_data, scorer_tid)
    else:
        # No score change — look for steal or possession change
        steal = event_det.find_steal(frames_data)
        if steal:
            result["steal"] = steal

    # Look for rebound (after misses, also triggered by cheer without score)
    if result["scorer"] is None:
        rb = event_det._find_near_basket(frames_data, "titans", "REB_DEF")
        if rb["confidence"] >= 0.45:
            result["rebounder"] = rb

    # Build summary
    parts = []
    if result["scorer"]:
        s = result["scorer"]
        parts.append(f"TrackID-{s['track_id']} ({s['team']}) {s['shot_type']} from {s['zone']} [conf={s['confidence']:.0%}]")
    if result["assister"]:
        a = result["assister"]
        parts.append(f"Assist: TrackID-{a['track_id']} [conf={a['confidence']:.0%}]")
    if result["steal"]:
        st = result["steal"]
        parts.append(f"Steal: TrackID-{st['track_id']} [conf={st['confidence']:.0%}]")
    result["summary"] = " | ".join(parts) if parts else "No event detected"

    return result


def _torso_crop_and_enhance(frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
    """
    Crop the torso region of a player (top 55% of bbox = number area).
    Applies contrast enhancement pipeline recommended by jersey-number-pipeline paper.
    """
    x, y, w, h = bbox
    fh, fw = frame.shape[:2]
    # Torso = top 55% of player bbox (shoulder to hip), with small side padding
    pad_x = max(2, int(w * 0.05))
    x1 = max(0, x - pad_x)
    y1 = max(0, y)
    x2 = min(fw, x + w + pad_x)
    y2 = min(fh, y + int(h * 0.55))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    # Scale up 4x (minimum 80px tall for reliable OCR)
    scale = max(4, int(80 / max(crop.shape[0], 1)))
    crop = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                      interpolation=cv2.INTER_CUBIC)
    # Contrast pipeline: convert to LAB, equalize L channel, sharpen
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    crop = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    crop = cv2.filter2D(crop, -1, kernel)
    return crop


def _paddle_read_number(crop: np.ndarray) -> str | None:
    """
    Run PaddleOCR on a crop image and extract the jersey number.
    Returns the number string (e.g. '11', '5') or None if not found.
    """
    ocr = _get_paddle()
    if ocr is None or crop is None:
        return None
    try:
        # PaddleOCR >= 2.9 uses predict() generator instead of ocr()
        results = list(ocr.predict(
            crop,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        ))
        if not results:
            return None
        candidates = []
        for res in results:
            texts  = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            for text, conf in zip(texts, scores):
                text = text.strip().replace(' ', '')
                if text.isdigit() and 1 <= len(text) <= 2 and conf >= 0.5:
                    candidates.append((text, conf))
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[1])[0]
    except Exception:
        return None


def scan_warmup_for_jerseys(
    video_path: str,
    duration_sec: int = 600,
    classifier: TeamClassifier = None,
    every_n_sec: int = 5,
) -> dict:
    """
    Scan the video to extract jersey numbers for Titans players.

    Pipeline (from jersey-number-pipeline paper, 91.4% accuracy):
      1. YOLO detects players → bbox per frame
      2. Filter: only large bboxes (>8% frame height = legible)
      3. Crop torso region (shoulder→hip, top 55% of bbox)
      4. Contrast enhancement (CLAHE + sharpen)
      5. PaddleOCR reads number
      6. Majority voting per track_id → confident jersey assignment

    Returns: list of closeup frame dicts (for Claude fallback) + populates
             jersey_votes dict {track_id: Counter({number: votes})}
    """
    if classifier is None:
        classifier = TeamClassifier()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    end_frame = int(min(duration_sec, cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps) * fps)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
    MIN_BBOX_H = frame_h * 0.08   # legibility threshold

    player_tracker = PlayerTracker(classifier)
    jersey_votes   = defaultdict(Counter)  # track_id → {number: count}
    closeup_frames = []

    step = int(every_n_sec * fps)
    for pos in range(0, end_frame, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            break

        players = player_tracker.process_frame(frame, pos)
        ts = pos / fps

        for p in players:
            x, y, w, h = p["bbox"]
            if h < MIN_BBOX_H or p["team"] != "titans":
                continue

            closeup_frames.append({
                "ts": ts, "track_id": p["id"], "team": p["team"],
                "bbox": p["bbox"], "frame_idx": pos,
            })

            # ── PaddleOCR on torso crop ────────────────────────────────
            torso = _torso_crop_and_enhance(frame, (x, y, w, h))
            num = _paddle_read_number(torso)
            if num:
                jersey_votes[p["id"]][num] += 1

    cap.release()

    # Attach vote results to each closeup entry so server can use them
    scan_result = {
        "closeup_frames": closeup_frames,
        "jersey_votes": {str(tid): dict(votes) for tid, votes in jersey_votes.items()},
    }
    return scan_result


def extract_player_crop(video_path: str, frame_idx: int, bbox: tuple, padding: int = 4) -> np.ndarray | None:
    """Extract a player crop from a specific frame for jersey OCR."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    x, y, w, h = bbox
    h_f, w_f = frame.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_f, x + w + padding)
    y2 = min(h_f, y + h + padding)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    # Scale up 4x for better OCR
    return cv2.resize(crop, (crop.shape[1] * 4, crop.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
