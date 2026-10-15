# -*- coding: utf-8 -*-
"""
Hand Cricket  -  single webcam, two-player, gesture-based.

Gesture system: rule-based finger counting (no ML model needed).
  Fist (0 fingers) = ready signal
  1 finger up      = 1 run
  2 fingers up     = 2 runs
  3 fingers up     = 3 runs
  4 fingers up     = 4 runs
  5 fingers up     = 6 runs

Ball flow
---------
  1. BALL ANNOUNCE  : "Ball N" shown for 1.8 s
  2. BATTER CAPTURE : 5-s countdown, batter shows gesture (fist -> open)
  3. LOOK AWAY      : "Bowler look away!" screen, SPACE to continue
  4. BOWLER CAPTURE : 5-s countdown, bowler shows gesture
  5. REVEAL         : both gestures shown + runs/OUT for 3 s
  repeat until 6 balls or OUT
"""

import cv2
import time
import numpy as np
import warnings
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe import ImageFormat

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE           = os.path.dirname(os.path.abspath(__file__))
LANDMARKER_PATH = os.path.join(_HERE, 'hand_landmarker.task')

# ── Constants ─────────────────────────────────────────────────────────────────
WIN        = "Hand Cricket"
CAM_INDEX  = 0
COUNTDOWN  = 5      # seconds per player per ball
REVEAL_DUR = 3.0    # seconds to show result
MAX_BALLS  = 6

# Maps finger count -> runs (5 fingers = 6 runs)
COUNT_TO_RUNS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6}
COUNT_TO_LABEL = {0: 'FIST', 1: '1', 2: '2', 3: '3', 4: '4', 5: '6'}

# ── Finger counting (pure geometry, no ML) ───────────────────────────────────
# Landmark indices:
#   Wrist=0
#   Thumb:  CMC=1 MCP=2 IP=3 TIP=4
#   Index:  MCP=5 PIP=6 DIP=7 TIP=8
#   Middle: MCP=9 PIP=10 DIP=11 TIP=12
#   Ring:   MCP=13 PIP=14 DIP=15 TIP=16
#   Pinky:  MCP=17 PIP=18 DIP=19 TIP=20

def count_fingers(lms):
    """
    Count extended fingers from MediaPipe hand landmarks.
    Returns (count 0-5, is_fist bool).
    Uses tip-vs-pip vertical check for index/middle/ring/pinky,
    and tip-vs-mcp horizontal check for thumb (works for both hands).
    Applies majority voting over the last few frames via the caller.
    """
    # y coordinates (image space: smaller y = higher on screen)
    y = [lm.y for lm in lms]
    x = [lm.x for lm in lms]

    fingers_up = 0

    # ── Thumb: compare tip x vs IP x (horizontal spread)
    # If tip is further from palm centre than IP joint → extended
    # Palm centre x is roughly the middle MCP (landmark 9)
    palm_cx = x[9]
    thumb_extended = abs(x[4] - palm_cx) > abs(x[3] - palm_cx)
    if thumb_extended:
        fingers_up += 1

    # ── Four fingers: tip y < pip y → extended (tip above pip in image)
    finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]   # (tip, pip)
    for tip_i, pip_i in finger_pairs:
        if y[tip_i] < y[pip_i]:
            fingers_up += 1

    is_fist = fingers_up == 0
    return fingers_up, is_fist


def gesture_label(count):
    """Human-readable label for display."""
    return COUNT_TO_LABEL.get(count, str(count))


def gesture_runs(count):
    """Runs value for a finger count."""
    return COUNT_TO_RUNS.get(count, 0)

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
W   = (255, 255, 255)
BLK = (10,  10,  10 )
GRN = (40,  210, 40 )
YLW = (0,   210, 210)
RED = (50,  50,  230)
BLU = (210, 150, 40 )
ORG = (0,   140, 255)
DRK = (18,  18,  18 )
CYN = (210, 210, 0  )

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),
    (9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]

# ── MediaPipe ─────────────────────────────────────────────────────────────────
def make_landmarker():
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.65,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def detect_gesture(frame, lmkr):
    """
    Returns (finger_count 0-5 | None, annotated_frame).
    None means no hand detected.
    """
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=ImageFormat.SRGB, data=rgb)
    res    = lmkr.detect(mp_img)
    if not res.hand_landmarks:
        return None, frame
    lms = res.hand_landmarks[0]
    h, fw = frame.shape[:2]
    pts = [(int(lm.x * fw), int(lm.y * h)) for lm in lms]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 180, 60), 2)
    for px, py in pts:
        cv2.circle(frame, (px, py), 5, W,   -1)
        cv2.circle(frame, (px, py), 5, GRN,  1)
    count, _ = count_fingers(lms)
    return count, frame


# ── Generic draw helpers ───────────────────────────────────────────────────────
def txt(img, text, pos, scale=0.75, color=W, thick=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, BLK, thick+2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def panel(img, x, y, pw, ph, color=DRK, alpha=0.80):
    ov = img.copy()
    cv2.rectangle(ov, (x, y), (x+pw, y+ph), color, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)
    cv2.rectangle(img, (x, y), (x+pw, y+ph), (70, 70, 70), 1)


def centered_txt(img, text, cy, scale=0.9, color=W, thick=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    IH, IW = img.shape[:2]
    txt(img, text, (IW//2 - tw//2, cy), scale, color, thick)


# ── Scoreboard (top strip) ─────────────────────────────────────────────────────
def draw_scoreboard(frame, batter, bowler, score, ball_num, inning, target=None):
    IH, IW = frame.shape[:2]
    panel(frame, 0, 0, IW, 60, (10, 10, 10), 0.88)
    cv2.line(frame, (0, 60), (IW, 60), (60, 60, 60), 1)

    # Left: inning + ball
    txt(frame, f"INNING {inning}   Ball {ball_num}/{MAX_BALLS}", (14, 40), 0.70, YLW)

    # Centre: score
    score_str = f"  {batter}:  {score} runs  "
    (sw, _), _ = cv2.getTextSize(score_str, cv2.FONT_HERSHEY_SIMPLEX, 0.80, 2)
    panel(frame, IW//2 - sw//2 - 8, 8, sw+16, 46, (0, 50, 0), 0.85)
    txt(frame, score_str, (IW//2 - sw//2, 42), 0.80, GRN, 2)

    # Right: bowler
    txt(frame, f"BOWL: {bowler}", (IW - 220, 40), 0.65, BLU)

    # Target bar (2nd inning)
    if target is not None:
        need   = max(0, target - score)
        t_str  = f"Target: {target}   Need: {need}"
        (tw2, _), _ = cv2.getTextSize(t_str, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        panel(frame, IW//2 - tw2//2 - 10, 60, tw2+20, 28, (0, 40, 80), 0.85)
        txt(frame, t_str, (IW//2 - tw2//2, 81), 0.65, CYN)


# ── Gesture cheat-sheet (bottom-left corner) ──────────────────────────────────
def draw_cheatsheet(frame):
    IH, IW = frame.shape[:2]
    lines = [
        ("GESTURES (fingers up):", YLW, 0.52),
        ("FIST  (0) = ready",      W,   0.46),
        ("  1 finger  =  1 run",   GRN, 0.46),
        ("  2 fingers =  2 runs",  GRN, 0.46),
        ("  3 fingers =  3 runs",  GRN, 0.46),
        ("  4 fingers =  4 runs",  GRN, 0.46),
        ("  5 fingers =  6 runs",  GRN, 0.46),
        ("SAME count  =  OUT!",    RED, 0.50),
    ]
    bh = len(lines) * 20 + 10
    panel(frame, 6, IH - bh - 8, 240, bh + 8, (10, 10, 10), 0.82)
    for i, (line, col, sc) in enumerate(lines):
        txt(frame, line, (12, IH - bh + i*20 + 4), sc, col, 1)


# ── Live gesture state badge (top-right during countdown) ─────────────────────
def draw_gesture_badge(frame, count, locked):
    """count and locked are integers (finger count 0-5) or None."""
    IH, IW = frame.shape[:2]
    bx, by, bw, bh = IW - 250, 68, 244, 70
    panel(frame, bx, by, bw, bh, (0, 30, 0), 0.85)
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), GRN, 1)

    if locked is not None:
        runs = gesture_runs(locked)
        lbl  = gesture_label(locked)
        txt(frame, "LOCKED IN:",         (bx+8,  by+24), 0.55, YLW, 1)
        txt(frame, lbl,                  (bx+8,  by+58), 1.1,  GRN, 2)
        txt(frame, f"= {runs} runs",     (bx+100, by+58), 0.65, W,   1)
    else:
        if count is None:
            txt(frame, "LIVE:",               (bx+8, by+24), 0.55, YLW, 1)
            txt(frame, "no hand",             (bx+8, by+58), 0.85, (120,120,120), 1)
        elif count == 0:
            txt(frame, "LIVE:  FIST",         (bx+8, by+24), 0.55, ORG, 1)
            txt(frame, "open hand now!",      (bx+8, by+52), 0.70, ORG, 2)
        else:
            runs = gesture_runs(count)
            lbl  = gesture_label(count)
            txt(frame, "LIVE:",               (bx+8,  by+24), 0.55, YLW, 1)
            txt(frame, lbl,                   (bx+8,  by+58), 1.1,  W,   2)
            txt(frame, f"= {runs} runs",      (bx+100, by+58), 0.65, (160,160,160), 1)


# ── Big countdown ring (right side) ───────────────────────────────────────────
def draw_countdown_ring(frame, secs_left, role_color):
    IH, IW = frame.shape[:2]
    cx, cy = IW - 390, IH // 2

    total   = float(COUNTDOWN)
    frac    = secs_left / total
    angle   = int(360 * frac)
    radius  = 55

    # Background ring
    cv2.circle(frame, (cx, cy), radius, (50, 50, 50), 8)
    # Progress arc
    arc_col = GRN if frac > 0.5 else YLW if frac > 0.25 else RED
    for a in range(0, angle, 3):
        rad = np.radians(a - 90)
        px  = int(cx + radius * np.cos(rad))
        py  = int(cy + radius * np.sin(rad))
        cv2.circle(frame, (px, py), 5, arc_col, -1)

    # Number inside
    num_s = str(int(np.ceil(secs_left))) if secs_left > 0 else "0"
    (nw, nh), _ = cv2.getTextSize(num_s, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
    txt(frame, num_s, (cx - nw//2, cy + nh//2), 1.6, arc_col, 3)

    txt(frame, "seconds", (cx - 35, cy + radius + 22), 0.50, W, 1)


# ── Player-turn header ─────────────────────────────────────────────────────────
def draw_turn_header(frame, player_name, role, role_color):
    IH, IW = frame.shape[:2]
    top = 62 if True else 90   # below scoreboard
    panel(frame, 0, top, IW, 48, (0, 0, 0), 0.80)
    cv2.line(frame, (0, top+48), (IW, top+48), role_color, 2)

    role_tag = f"[ {role.upper()} ]"
    txt(frame, role_tag,     (12,     top+32), 0.75, role_color, 2)
    txt(frame, player_name,  (160,    top+32), 0.90, W,          2)
    txt(frame, "- show your gesture!", (160 + int(len(player_name)*13.5), top+32),
        0.65, YLW, 1)


# ── Full-screen "look away" / "get ready" screen ──────────────────────────────
def look_away_screen(cap, waiting_player, active_player, role):
    """Solid colour screen so waiting player can't see active player's gesture."""
    IH, IW = 720, 1280
    bg = np.full((IH, IW, 3), (10, 10, 30), dtype=np.uint8)

    panel(bg, IW//2 - 300, IH//2 - 120, 600, 240, (15, 15, 50), 0.96)
    cv2.rectangle(bg, (IW//2-300, IH//2-120), (IW//2+300, IH//2+120), YLW, 2)

    centered_txt(bg, f"{waiting_player},",        IH//2 - 65, 1.0,  YLW, 2)
    centered_txt(bg, "please look away!",          IH//2 - 20, 0.90, W,   2)
    centered_txt(bg, f"{active_player} ({role}),", IH//2 + 35, 0.80, GRN, 2)
    centered_txt(bg, "get ready to show gesture.", IH//2 + 70, 0.75, W,   1)
    centered_txt(bg, "Press SPACE when ready",     IH//2 + 110,0.65, CYN, 1)

    cv2.imshow(WIN, bg)
    while True:
        k = cv2.waitKey(30) & 0xFF
        if k == ord(' ') or k == 13:   # SPACE or ENTER
            break


# ── Announce ball screen ───────────────────────────────────────────────────────
def announce_ball(cap, ball_num, batter, bowler, score, inning, target):
    end = time.time() + 1.8
    while time.time() < end:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        IH, IW = frame.shape[:2]

        # Dim
        ov = frame.copy(); ov[:] = BLK
        cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)

        draw_scoreboard(frame, batter, bowler, score, ball_num, inning, target)

        panel(frame, IW//2 - 200, IH//2 - 55, 400, 110, (10, 30, 10), 0.88)
        cv2.rectangle(frame, (IW//2-200, IH//2-55),(IW//2+200, IH//2+55), GRN, 2)
        centered_txt(frame, f"Ball  {ball_num}", IH//2 - 10, 1.4, YLW, 3)
        draw_cheatsheet(frame)
        cv2.imshow(WIN, frame)
        cv2.waitKey(1)


# ── Core: capture one gesture with countdown ──────────────────────────────────
def capture_gesture(cap, lmkr, player, role, role_color,
                    batter, bowler, score, ball_num, inning, target):
    """
    Returns locked finger count (int 0-5). 0 = fist.
    Protocol: show fist (count=0) -> open to desired number -> locked.
    Can re-lock multiple times; last one counts.
    """
    deadline   = time.time() + COUNTDOWN
    locked     = None   # int finger count
    fist_seen  = False
    last_count = None   # last raw detection

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        count, frame = detect_gesture(frame, lmkr)   # int 0-5 or None

        # Locking logic: fist (0) -> any non-fist -> lock it
        if count == 0:
            fist_seen = True
        elif fist_seen and count is not None and count > 0:
            locked    = count
            fist_seen = False
        last_count = count

        remaining = max(0.0, deadline - time.time())

        IH, IW = frame.shape[:2]

        draw_scoreboard(frame, batter, bowler, score, ball_num, inning, target)
        draw_turn_header(frame, player, role, role_color)
        draw_countdown_ring(frame, remaining, role_color)
        draw_gesture_badge(frame, last_count, locked)

        # Step guide at bottom centre
        if not fist_seen and locked is None:
            step_txt, step_col = "STEP 1: Make a CLOSED FIST", ORG
        elif fist_seen:
            step_txt, step_col = "STEP 2: Open hand to show your number", YLW
        else:
            step_txt, step_col = "LOCKED!  Fist again to change", GRN

        panel(frame, IW//2 - 260, IH - 80, 520, 34, (10, 10, 10), 0.80)
        centered_txt(frame, step_txt, IH - 55, 0.72, step_col, 2)

        draw_cheatsheet(frame)

        # Centre flash when locked
        if locked is not None and not fist_seen:
            lbl  = gesture_label(locked)
            runs = gesture_runs(locked)
            lx, ly = IW//2 - 150, IH//2 - 35
            panel(frame, lx, ly, 300, 60, (0, 60, 0), 0.75)
            centered_txt(frame, f"LOCKED:  {lbl}  ({runs} runs)", IH//2 + 5, 0.90, GRN, 2)

        cv2.imshow(WIN, frame)
        cv2.waitKey(1)

        if remaining <= 0:
            break

    # If nothing locked, use last detected count (even fist = 0)
    if locked is None:
        locked = last_count if last_count is not None else 0
    return locked


# ── Result reveal screen ──────────────────────────────────────────────────────
def reveal_screen(cap, batter, bowler, b_count, w_count, runs, out, score, inning, ball_num, target):
    b_runs = gesture_runs(b_count)
    w_runs = gesture_runs(w_count)
    b_lbl  = gesture_label(b_count)
    w_lbl  = gesture_label(w_count)
    end    = time.time() + REVEAL_DUR

    while time.time() < end:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        IH, IW = frame.shape[:2]

        # Dim background
        ov = np.zeros_like(frame)
        cv2.addWeighted(ov, 0.50, frame, 0.50, 0, frame)

        draw_scoreboard(frame, batter, bowler, score, ball_num, inning, target)

        # Result card
        cx, cy   = IW//2, IH//2
        cw, ch   = 520, 260
        px, py   = cx - cw//2, cy - ch//2
        bg_col   = (40, 10, 10) if out else (10, 40, 10)
        border   = RED if out else GRN
        panel(frame, px, py, cw, ch, bg_col, 0.92)
        cv2.rectangle(frame, (px, py), (px+cw, py+ch), border, 3)

        # Header
        header = "  W I C K E T !  " if out else "  R U N S  "
        h_col  = RED if out else YLW
        centered_txt(frame, header, py + 45, 1.1, h_col, 3)

        # Divider
        cv2.line(frame, (px+20, py+60), (px+cw-20, py+60), (80,80,80), 1)

        # Batter row
        b_str = f"[BAT] {batter}:  {b_lbl} fingers  ({b_runs} runs)"
        txt(frame, b_str, (px + 22, py + 110), 0.75, GRN, 2)

        # Bowler row
        w_str = f"[BOW] {bowler}:  {w_lbl} fingers  ({w_runs} runs)"
        txt(frame, w_str, (px + 22, py + 155), 0.75, BLU, 2)

        # Big verdict
        cv2.line(frame, (px+20, py+170), (px+cw-20, py+170), (80,80,80), 1)
        if out:
            centered_txt(frame, "OUT!  Both showed the same number.", py + 220, 0.72, RED, 2)
        else:
            run_col = YLW if runs >= 4 else GRN if runs > 0 else (120,120,120)
            run_msg = f"+{runs} run{'s' if runs != 1 else ''}" if runs > 0 else "Dot ball  (0 runs)"
            centered_txt(frame, run_msg, py + 220, 0.90, run_col, 2)

        cv2.imshow(WIN, frame)
        cv2.waitKey(1)


# ── Inning end splash ─────────────────────────────────────────────────────────
def inning_end_screen(cap, batter, score, inning):
    end = time.time() + 3.5
    while time.time() < end:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        IH, IW = frame.shape[:2]
        ov = np.zeros_like(frame)
        cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)

        panel(frame, IW//2-260, IH//2-90, 520, 180, (15, 15, 40), 0.93)
        cv2.rectangle(frame,(IW//2-260,IH//2-90),(IW//2+260,IH//2+90), YLW, 2)
        centered_txt(frame, f"Inning {inning} complete",  IH//2 - 40, 1.0, YLW, 2)
        centered_txt(frame, f"{batter}:  {score} runs",   IH//2 + 15, 0.95, GRN, 2)
        centered_txt(frame, "Switching innings...",        IH//2 + 60, 0.65, W,   1)
        cv2.imshow(WIN, frame)
        cv2.waitKey(1)


# ── Final result screen ───────────────────────────────────────────────────────
def final_screen(cap, p1, p2, s1, s2):
    if   s1 > s2: winner, detail = p1, f"wins by {s1-s2} run{'s' if s1-s2!=1 else ''}"
    elif s2 > s1: winner, detail = p2, f"wins by {s2-s1} run{'s' if s2-s1!=1 else ''}"
    else:         winner, detail = "TIE", "Match Tied!"

    while True:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame = cv2.flip(frame, 1)
        IH, IW = frame.shape[:2]

        ov = frame.copy(); ov[:] = (10,10,10)
        cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)

        panel(frame, IW//2-310, 100, 620, 420, (12,12,30), 0.94)
        cv2.rectangle(frame,(IW//2-310,100),(IW//2+310,520), YLW, 3)

        centered_txt(frame, "MATCH  OVER",              150, 1.3, YLW, 3)
        cv2.line(frame, (IW//2-280,175),(IW//2+280,175),(80,80,80),1)

        # Scores
        panel(frame, IW//2-130, 190, 260, 50, (0,40,0), 0.80)
        centered_txt(frame, f"{p1}:  {s1} runs", 225, 0.90, GRN, 2)
        panel(frame, IW//2-130, 250, 260, 50, (0,0,40), 0.80)
        centered_txt(frame, f"{p2}:  {s2} runs", 285, 0.90, BLU, 2)

        cv2.line(frame,(IW//2-280,315),(IW//2+280,315),(80,80,80),1)

        if winner == "TIE":
            centered_txt(frame, "MATCH  TIED!", 385, 1.4, ORG, 3)
        else:
            centered_txt(frame, f"{winner}", 370, 1.3, YLW, 3)
            centered_txt(frame, detail,       420, 0.85, W,   2)

        centered_txt(frame, "Press any key to exit", 490, 0.60, (140,140,140), 1)
        cv2.imshow(WIN, frame)
        if cv2.waitKey(80) & 0xFF != 255:
            break


# ── Toss screen ───────────────────────────────────────────────────────────────
def toss_screen(cap, p1, p2):
    result = np.random.choice(['HEADS', 'TAILS'])
    phase  = 'call'
    call   = None
    winner = None

    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        IH, IW = frame.shape[:2]

        ov = frame.copy(); ov[:] = BLK
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

        panel(frame, IW//2-280, 80, 560, 420, (12,12,12), 0.93)
        cv2.rectangle(frame,(IW//2-280,80),(IW//2+280,500), YLW, 2)

        centered_txt(frame, "TOSS",               135, 1.5, YLW, 3)
        cv2.line(frame,(IW//2-250,160),(IW//2+250,160),(80,80,80),1)

        if phase == 'call':
            centered_txt(frame, f"{p1},",                   215, 0.80, W,   2)
            centered_txt(frame, "press  H  for HEADS",      260, 0.78, GRN, 2)
            centered_txt(frame, "press  T  for TAILS",      300, 0.78, BLU, 2)

        elif phase == 'flip':
            centered_txt(frame, f"Result:  {result}",       220, 1.0,  GRN, 2)
            w_txt = f"{p1 if call==result.lower() else p2} wins the toss!"
            centered_txt(frame, w_txt,                      275, 0.85, YLW, 2)
            centered_txt(frame, "Press any key to continue",340, 0.65, W,   1)

        elif phase == 'bat_bowl':
            centered_txt(frame, f"{winner},",               215, 0.85, YLW, 2)
            centered_txt(frame, "press  B  to  BAT",        260, 0.80, GRN, 2)
            centered_txt(frame, "press  W  to  BOWL",       305, 0.80, BLU, 2)

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(30) & 0xFF

        if phase == 'call':
            if key == ord('h'):
                call   = 'heads'
                winner = p1 if result == 'HEADS' else p2
                phase  = 'flip'
            elif key == ord('t'):
                call   = 'tails'
                winner = p1 if result == 'TAILS' else p2
                phase  = 'flip'

        elif phase == 'flip':
            if key != 255:
                phase = 'bat_bowl'

        elif phase == 'bat_bowl':
            if key == ord('b'):
                loser = p2 if winner == p1 else p1
                return winner, loser, 'bat'
            elif key == ord('w'):
                loser = p2 if winner == p1 else p1
                return winner, loser, 'bowl'


# ── Play one inning ───────────────────────────────────────────────────────────
def play_inning(cap, lmkr, batter, bowler, inning_num, target=None):
    score = 0

    for ball in range(1, MAX_BALLS + 1):

        # Ball announce
        announce_ball(cap, ball, batter, bowler, score, inning_num, target)

        # ── Batter turn ──────────────────────────────────────────────────────
        look_away_screen(cap, bowler, batter, "BATTER")
        b_count = capture_gesture(cap, lmkr, batter, "BATTER", GRN,
                                  batter, bowler, score, ball, inning_num, target)

        # ── Bowler turn ──────────────────────────────────────────────────────
        look_away_screen(cap, batter, bowler, "BOWLER")
        w_count = capture_gesture(cap, lmkr, bowler, "BOWLER", BLU,
                                  batter, bowler, score, ball, inning_num, target)

        # ── Evaluate ─────────────────────────────────────────────────────────
        b_runs = gesture_runs(b_count)
        w_runs = gesture_runs(w_count)
        out    = (b_runs == w_runs)
        runs   = 0 if out else b_runs

        # ── Reveal ───────────────────────────────────────────────────────────
        new_score = score if out else score + runs
        reveal_screen(cap, batter, bowler, b_count, w_count,
                      runs, out, new_score, inning_num, ball, target)

        if out:
            break

        score += runs

        # Chase success
        if target is not None and score >= target:
            break

    return score, out


# ── Name entry (terminal only, once) ─────────────────────────────────────────
def get_names():
    print("\n" + "="*46)
    print("          HAND CRICKET")
    print("="*46)
    print("One webcam. Players take turns each ball.")
    print("All game screens shown on the camera window.\n")
    p1 = input("Player 1 name: ").strip() or "Player 1"
    p2 = input("Player 2 name: ").strip() or "Player 2"
    return p1, p2


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p1_name, p2_name = get_names()

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    lmkr = make_landmarker()

    # Toss
    first_bat, first_bowl, choice = toss_screen(cap, p1_name, p2_name)
    if choice == 'bowl':
        first_bat, first_bowl = first_bowl, first_bat

    # Inning 1
    score1, _ = play_inning(cap, lmkr, first_bat, first_bowl,
                             inning_num=1, target=None)
    inning_end_screen(cap, first_bat, score1, 1)

    # Inning 2
    score2, _ = play_inning(cap, lmkr, first_bowl, first_bat,
                             inning_num=2, target=score1 + 1)
    inning_end_screen(cap, first_bowl, score2, 2)

    # Final
    final_screen(cap, first_bat, first_bowl, score1, score2)

    lmkr.close()
    cap.release()
    cv2.destroyAllWindows()
