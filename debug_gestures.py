# -*- coding: utf-8 -*-
"""
Live gesture debugger.
Run this, show your hand, and watch what the model predicts in real time.
Also collects new training samples so you can retrain on YOUR hand.

Controls
--------
  Press 1 2 3 4 6 F  -> save current frame landmarks as that label
                         (F = fist / closed)
  Press C            -> clear all collected samples
  Press T            -> train a NEW model on collected samples + original data
  Press Q            -> quit
"""

import cv2, pickle, csv, os, time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe import ImageFormat
import warnings
warnings.filterwarnings("ignore")

_HERE           = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(_HERE, 'hand_model.pkl')
LANDMARKER_PATH = os.path.join(_HERE, 'hand_landmarker.task')
ORIG_CSV        = os.path.join(_HERE, 'cricket_dataset1.csv')
NEW_CSV         = os.path.join(_HERE, 'new_samples.csv')

with open(MODEL_PATH, 'rb') as f:
    rf = pickle.load(f)

opts = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)
lmkr = mp_vision.HandLandmarker.create_from_options(opts)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),
    (9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]

KEY_TO_LABEL = {
    ord('1'): 'one',
    ord('2'): 'two',
    ord('3'): 'three',
    ord('4'): 'four',
    ord('6'): 'six',
    ord('f'): 'closed',
}

collected   = []   # list of (label, flat_features)
status_msg  = ""
status_time = 0
last_flat   = None
last_pred   = None

# Confidence-style: get all class probabilities
CLASSES = rf.classes_

def draw_bar(img, label, prob, y, bar_color):
    bw = int(prob * 300)
    cv2.rectangle(img, (10, y), (10 + bw, y + 22), bar_color, -1)
    cv2.rectangle(img, (10, y), (310, y + 22), (60,60,60), 1)
    cv2.putText(img, f"{label:<8} {prob*100:5.1f}%",
                (320, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow("Gesture Debugger", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture Debugger", 1280, 720)

print("\nGesture Debugger open.")
print("Show your hand and watch predictions.")
print("Keys: 1 2 3 4 6 F = save sample | C = clear | T = retrain | Q = quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    IH, IW = frame.shape[:2]

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=ImageFormat.SRGB, data=rgb)
    result = lmkr.detect(mp_img)

    probs    = None
    pred     = None
    last_flat = None

    if result.hand_landmarks:
        lms = result.hand_landmarks[0]
        h, fw = frame.shape[:2]
        pts = [(int(lm.x * fw), int(lm.y * h)) for lm in lms]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 180, 60), 2)
        for px, py in pts:
            cv2.circle(frame, (px, py), 5, (255,255,255), -1)
            cv2.circle(frame, (px, py), 5, (40, 210, 40), 1)

        last_flat = ([lm.x for lm in lms] +
                     [lm.y for lm in lms] +
                     [lm.z for lm in lms])
        probs = rf.predict_proba([last_flat])[0]
        pred  = CLASSES[np.argmax(probs)]
        last_pred = pred

    # ── Right panel: probability bars ────────────────────────────────────────
    panel_x = IW - 430
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x - 10, 0), (IW, IH), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "PREDICTIONS", (panel_x, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,210,210), 2)
    cv2.putText(frame, "(confidence per class)", (panel_x, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140,140,140), 1)

    BAR_COLORS = {
        'closed': (80,80,80),
        'one':    (40,160,40),
        'two':    (40,180,80),
        'three':  (40,200,140),
        'four':   (40,160,220),
        'six':    (40,80,220),
    }

    if probs is not None:
        sorted_idx = np.argsort(probs)[::-1]
        for rank, idx in enumerate(sorted_idx):
            lbl = CLASSES[idx]
            p   = probs[idx]
            y   = 80 + rank * 40
            col = BAR_COLORS.get(lbl, (100,100,100))
            # highlight winner
            if lbl == pred:
                col = tuple(min(255, c + 80) for c in col)

            bw = int(p * 300)
            cv2.rectangle(frame, (panel_x, y), (panel_x + bw, y + 28), col, -1)
            cv2.rectangle(frame, (panel_x, y), (panel_x + 300, y + 28), (60,60,60), 1)
            cv2.putText(frame, f"{lbl:<8} {p*100:5.1f}%",
                        (panel_x + 308, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (255,255,200) if lbl == pred else (160,160,160), 1)

    # ── Top: big prediction ───────────────────────────────────────────────────
    if pred:
        cv2.rectangle(frame, (0, 0), (panel_x - 20, 60), (10,10,10), -1)
        cv2.putText(frame, f"DETECTED:  {pred.upper()}",
                    (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 5)
        col = (40,210,40) if pred != 'closed' else (0,200,200)
        cv2.putText(frame, f"DETECTED:  {pred.upper()}",
                    (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
    else:
        cv2.putText(frame, "No hand detected", (12, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120,120,120), 2)

    # ── Bottom: instructions + sample count ──────────────────────────────────
    instrs = [
        "1/2/3/4/6 = save sample with that label",
        "F = save as FIST/closed    C = clear samples",
        "T = retrain model on your samples    Q = quit",
        f"Samples collected: {len(collected)}",
    ]
    for i, ln in enumerate(instrs):
        col = (0,210,210) if i == 3 else (160,160,160)
        cv2.putText(frame, ln, (12, IH - 80 + i*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)

    # ── Status message ────────────────────────────────────────────────────────
    if status_msg and time.time() - status_time < 2.5:
        cv2.putText(frame, status_msg, (12, IH - 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 100), 2)

    # ── Finger state indicators ───────────────────────────────────────────────
    if result.hand_landmarks:
        lms_raw = result.hand_landmarks[0]
        y_vals  = [lm.y for lm in lms_raw]
        # MCP vs tip for 4 fingers
        finger_pairs = [(5,8,'Idx'),(9,12,'Mid'),(13,16,'Rng'),(17,20,'Pnk')]
        for fi, (mcp, tip, name) in enumerate(finger_pairs):
            extended = y_vals[tip] < y_vals[mcp]
            color    = (40, 210, 40) if extended else (60, 60, 60)
            state    = "UP" if extended else "dn"
            fx       = 12 + fi * 80
            cv2.putText(frame, name, (fx, IH - 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180,180,180), 1)
            cv2.putText(frame, state, (fx, IH - 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)

    cv2.imshow("Gesture Debugger", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key in KEY_TO_LABEL and last_flat is not None:
        label = KEY_TO_LABEL[key]
        collected.append((label, last_flat))
        status_msg  = f"Saved sample for '{label}'  (total {len(collected)})"
        status_time = time.time()
        print(status_msg)

    elif key == ord('c'):
        collected = []
        status_msg  = "Cleared all samples."
        status_time = time.time()
        print(status_msg)

    elif key == ord('t'):
        if len(collected) < 10:
            status_msg  = "Need at least 10 samples first! Keep collecting."
            status_time = time.time()
        else:
            print(f"\nRetraining on {len(collected)} new samples + original data...")
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score

            # Load original data
            orig_X, orig_y = [], []
            with open(ORIG_CSV) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    feat = ([float(row[f'x{i}']) for i in range(21)] +
                            [float(row[f'y{i}']) for i in range(21)] +
                            [float(row[f'z{i}']) for i in range(21)])
                    orig_X.append(feat)
                    orig_y.append(row['label'])

            # Add new samples (weighted 5x so your hand dominates)
            new_X = [s[1] for s in collected] * 5
            new_y = [s[0] for s in collected] * 5

            X = np.array(orig_X + new_X)
            y = np.array(orig_y + new_y)

            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=42)
            new_rf = RandomForestClassifier(n_estimators=150, random_state=42)
            new_rf.fit(X_tr, y_tr)
            acc = accuracy_score(y_te, new_rf.predict(X_te))
            print(f"New model accuracy: {acc*100:.1f}%")

            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(new_rf, f)
            rf = new_rf
            # Save new samples to CSV for reference
            with open(NEW_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                cols = ([f'x{i}' for i in range(21)] +
                        [f'y{i}' for i in range(21)] +
                        [f'z{i}' for i in range(21)] + ['label'])
                writer.writerow(cols)
                for lbl, feat in collected:
                    writer.writerow(feat + [lbl])

            status_msg  = f"Retrained! Accuracy: {acc*100:.1f}%  Model saved."
            status_time = time.time()
            print(status_msg)

lmkr.close()
cap.release()
cv2.destroyAllWindows()
print("Debugger closed.")
