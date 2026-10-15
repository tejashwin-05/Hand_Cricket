# Hand Cricket — Full Project Context

Use this document to generate a PowerPoint presentation for the Hand Cricket project.

---

## Project Title
**Hand Cricket** — A Gesture-Controlled Cricket Game using Computer Vision

---

## One-Line Summary
A two-player hand gesture cricket game that runs entirely through a webcam — no keyboard, no controller, just your fingers.

---

## Project Category
- Domain: Image Processing & Computer Vision (IPCV)
- Type: Real-time Interactive Application
- Platform: Desktop (Windows/Linux/macOS), Single Webcam

---

## Problem Statement
Traditional hand cricket is played physically between two people. This project digitizes that experience using computer vision — allowing two players to play hand cricket through a webcam by recognizing finger gestures in real time, with a full game UI, scoreboard, toss, and innings system.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.x |
| Hand Detection | MediaPipe Hand Landmarker (Tasks API) |
| Gesture Recognition | Rule-based finger counting (geometric) |
| Video Capture & Display | OpenCV (cv2) |
| Numerical Processing | NumPy |
| ML Model (initial) | Scikit-learn Random Forest (trained on 3,600 samples) |
| Dataset | Custom CSV — 21 hand landmarks x (x,y,z) x 6 classes |

---

## System Architecture

```
Webcam Feed
    |
    v
MediaPipe Hand Landmarker
    |  (detects 21 landmarks per hand)
    v
Finger Counter (geometric)
    |  tip_y < pip_y  finger extended
    |  count: 0-5
    v
Game Logic (state machine)
    |
    v
OpenCV Game UI (1280x720 window)
```

---

## How Gesture Recognition Works

MediaPipe detects 21 landmarks on the hand. Each landmark has (x, y, z) coordinates normalized to the image.

**Finger counting logic (pure geometry — no ML):**
- For Index, Middle, Ring, Pinky: if `tip_y < pip_y` the finger is extended upward
- For Thumb: if `|tip_x - palm_centre_x| > |ip_x - palm_centre_x|` the thumb is spread

**Gesture to Runs mapping:**

| Fingers Up | Runs Scored |
|---|---|
| 0 (Fist) | Ready signal |
| 1 | 1 run |
| 2 | 2 runs |
| 3 | 3 runs |
| 4 | 4 runs |
| 5 (all) | 6 runs |
| Same count as bowler | OUT |

---

## Game Flow (State Machine)

```
Start
  -> Enter Player Names (terminal)
       -> TOSS SCREEN (H/T key press)
             -> Bat/Bowl choice (B/W key)
                   -> INNING 1
                   |     -> For each ball (max 6):
                   |           1. Ball Announce (1.8s)
                   |           2. "Bowler look away" screen  SPACE
                   |           3. Batter 5-second countdown + gesture capture
                   |           4. "Batter look away" screen  SPACE
                   |           5. Bowler 5-second countdown + gesture capture
                   |           6. Result Reveal (3s): OUT or +N runs
                   |     -> Inning End Screen
                   -> INNING 2 (with target)
                         -> Same ball flow
                               -> FINAL RESULT SCREEN
```

---

## Key Features

1. **Full game UI inside OpenCV window** — no terminal interaction during gameplay
2. **Live gesture badge** — shows what the camera currently sees in real time
3. **Countdown ring** — animated 5-second timer per player, changes colour (green to yellow to red)
4. **Look Away screen** — prevents the waiting player from seeing the other's gesture
5. **Step-by-step guide** — "Show fist, Open hand" instructions on screen every ball
6. **Toss system** — coin flip with H/T keyboard, bat/bowl choice
7. **Scoreboard** — always visible: inning, ball count, score, bowler, target (2nd inning)
8. **Result reveal** — shows both gestures side by side with verdict (OUT / +N runs)
9. **Inning transition** — 3.5s splash screen between innings
10. **Final match result** — winner display with run margin

---

## Dataset

- File: `cricket_dataset1.csv`
- Samples: 3,600 total (600 per class)
- Classes: `one`, `two`, `three`, `four`, `six`, `closed`
- Features: 63 per sample — `x0..x20`, `y0..y20`, `z0..z20` (all-x then all-y then all-z order)
- Source: Custom collected using MediaPipe on real hands

**Note:** The final version replaced the ML classifier with a rule-based geometric finger counter because the trained model was not generalizing well to different hands and lighting conditions. The geometric approach is deterministic and works for any person.

---

## Challenges Faced and Solutions

| Challenge | Solution |
|---|---|
| mediapipe 1.x dropped `mp.solutions` API | Migrated to `mp.tasks.vision.HandLandmarker` + downloaded `.task` model file |
| Only one webcam available, not two | Redesigned to single-cam turn-based mode with look-away screen |
| Game felt like terminal input, not a game | Full OpenCV UI with all screens, countdowns, results inside camera window |
| ML model only predicted 0 or 4 | Replaced Random Forest with pure geometric finger counting |
| Feature order mismatch in ML model | Training data was all-x then all-y then all-z but inference sent x,y,z per landmark — fixed extraction order |
| sklearn version mismatch on load | Retrained model fresh with installed version (99.7% accuracy on dataset) |

---

## Files in the Project

| File | Purpose |
|---|---|
| `app.py` | Main game application — entire game UI and logic |
| `hand_landmarker.task` | MediaPipe pre-trained hand landmark model file |
| `hand_model.pkl` | Trained Random Forest classifier (kept for reference) |
| `cricket_dataset1.csv` | 3,600-sample hand gesture dataset |
| `debug_gestures.py` | Developer tool — live gesture debugger with confidence bars, sample collector, and retrainer |
| `requirements.txt` | opencv-python, mediapipe==0.10.35, numpy, scikit-learn |

---

## How to Run

```bash
# 1. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the game
python app.py
```

---

## Key Learnings

- MediaPipe's landmark geometry is robust enough for finger counting without any ML model
- Gesture datasets collected by one person generalize poorly to others — geometric rules are more reliable for simple finger counting
- OpenCV works well as a real-time game UI engine
- Single-webcam turn-based design is a practical solution when hardware is limited
- Feature order in ML pipelines must exactly match between training and inference

---

## Suggested PPT Slide Structure

1. **Title Slide** — Hand Cricket: Gesture-Controlled Cricket Game
2. **Problem Statement** — Why and what we built
3. **Tech Stack** — table with all components
4. **System Architecture** — pipeline diagram
5. **How It Works** — gesture recognition and finger counting logic
6. **Gesture Reference Table** — fingers up to runs mapping
7. **Game Flow** — state machine / screenshots of each screen
8. **Key Features** — bullet list with UI highlights
9. **Dataset** — CSV stats, classes, how it was collected
10. **Challenges and Solutions** — table
11. **Demo / Screenshots**
12. **Learnings and Conclusion**
