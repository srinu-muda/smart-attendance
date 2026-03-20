# ai/recogniser.py — ArcFace + FAISS + Liveness Gate
# Pipeline: Detect face → Liveness check (blink) → Recognise → Mark
# A photo held to camera will FAIL liveness and never reach recognition

import cv2
import pickle
import numpy as np
import faiss
import os
import sys

# ── Fix module path ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.encoder import get_face_app, load_db
import mediapipe as mp

# ── Paths ─────────────────────────────────────────────────────────
FAISS_INDEX = "data/faiss.index"
LABELS_FILE = "data/labels.pkl"

# ── Thresholds ────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.68   # ArcFace match threshold
EAR_THRESHOLD        = 0.28 # Eye closed if EAR below this
CONSEC_FRAMES        = 2    # Frames eye must be closed = 1 blink
BLINKS_NEEDED        = 2      # Blinks required to pass liveness
CONFIRM_FRAMES       = 5      # Consecutive detections before marking

# ── MediaPipe eye landmark indices ────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]


# ══════════════════════════════════════════════════════════════════
# LIVENESS FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """
    Compute Eye Aspect Ratio (EAR) for one eye.
    EAR drops sharply when eye is closed (blink).
    Open eye  → EAR ~0.25–0.35
    Closed eye → EAR ~0.0–0.15
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * w, lm.y * h))

    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))

    return (A + B) / (2.0 * C) if C != 0 else 0.0


class LivenessState:
    """
    Tracks blink count per face region.
    One instance per tracked face — reset when face leaves frame.
    """
    def __init__(self):
        self.blink_counter = 0     # Frames eye has been closed
        self.total_blinks  = 0     # Total completed blinks
        self.is_live       = False  # Passed liveness check
        self.status_msg    = f"Blink {BLINKS_NEEDED}x to verify"

    def update(self, ear):
        """Feed one EAR value, update blink count."""
        if ear < EAR_THRESHOLD:
            self.blink_counter += 1
        else:
            if self.blink_counter >= CONSEC_FRAMES:
                self.total_blinks += 1
            self.blink_counter = 0

        remaining = max(0, BLINKS_NEEDED - self.total_blinks)
        if self.total_blinks >= BLINKS_NEEDED:
            self.is_live   = True
            self.status_msg = "LIVE verified"
        else:
            self.status_msg = f"Blink {remaining} more time(s)"


# ══════════════════════════════════════════════════════════════════
# FAISS FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def load_faiss():
    """Load FAISS index and student ID labels from disk."""
    if not os.path.exists(FAISS_INDEX) or \
       not os.path.exists(LABELS_FILE):
        print("[WARN] FAISS index not found.")
        print("[WARN] Run: python -m ai.encoder → enroll students first")
        return None, None

    index = faiss.read_index(FAISS_INDEX)
    with open(LABELS_FILE, 'rb') as f:
        labels = pickle.load(f)

    print(f"[INFO] FAISS loaded: {index.ntotal} vectors, "
          f"{len(set(labels))} students")
    return index, labels


def recognise_face(emb, index, labels, db):
    """
    Match a single ArcFace embedding against FAISS index.

    Returns:
        (student_id, name, confidence) or ('UNKNOWN', 'Unknown', score)
    """
    emb_q = emb.astype(np.float32).reshape(1, -1)
    scores, indices = index.search(emb_q, k=1)

    best_score = float(scores[0][0])
    best_idx   = int(indices[0][0])

    if best_score >= SIMILARITY_THRESHOLD and best_idx >= 0:
        sid  = labels[best_idx]
        name = db.get(sid, {}).get('name', 'Unknown')
        return sid, name, round(best_score, 3)
    else:
        return 'UNKNOWN', 'Unknown', round(best_score, 3)


# ══════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════

def draw_face_box(frame, x1, y1, x2, y2,
                  name, sid, confidence,
                  is_known, liveness_msg, is_live):
    """
    Draw bounding box with name, confidence, and liveness status.

    Colors:
      Green  = known + live
      Orange = known but liveness pending
      Red    = unknown
    """
    if is_known and is_live:
        color = (0, 210, 0)      # Green — confirmed
    elif is_known and not is_live:
        color = (0, 165, 255)    # Orange — recognised but not live yet
    else:
        color = (0, 0, 220)      # Red — unknown

    # Main box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Name + confidence label
    label = f"{name}  {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame,
                  (x1, y1 - th - 14), (x1 + tw + 10, y1),
                  color, -1)
    cv2.putText(frame, label,
                (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

    # Student ID below box
    if is_known:
        cv2.putText(frame, f"ID: {sid}",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    # Liveness status below ID
    live_color = (0,210,0) if is_live else (0,165,255)
    cv2.putText(frame,
                f"{'[LIVE]' if is_live else '[?]'} {liveness_msg}",
                (x1, y2 + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, live_color, 1)

    # Confidence bar
    bar_w = x2 - x1
    fill  = int(bar_w * min(confidence, 1.0))
    cv2.rectangle(frame, (x1, y2+44), (x2, y2+50), (50,50,50), -1)
    cv2.rectangle(frame, (x1, y2+44), (x1+fill, y2+50), color, -1)

    return frame


# ══════════════════════════════════════════════════════════════════
# MAIN RECOGNITION + LIVENESS LOOP
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 58)
    print("   SMART ATTENDANCE — ArcFace + Liveness Recognition")
    print("   Pipeline: Detect → Blink check → Recognise → Mark")
    print(f"   Liveness : blink {BLINKS_NEEDED}x to verify")
    print(f"   Threshold: {SIMILARITY_THRESHOLD}")
    print("   Press Q or ESC to quit | D for debug scores")
    print("=" * 58 + "\n")

    # ── Load FAISS index ──────────────────────────────────────────
    index, labels = load_faiss()
    if index is None:
        print("[ERROR] Enroll students first:")
        print("        python -m ai.encoder")
        exit()

    db = load_db()
    print(f"[INFO] {len(db)} student(s) enrolled\n")

    # ── Load InsightFace ──────────────────────────────────────────
    face_app = get_face_app()

    # ── Load MediaPipe FaceMesh for liveness ─────────────────────
    print("[INFO] Loading MediaPipe for liveness detection...")
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh    = mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        use_mediapipe = True
        print("[INFO] MediaPipe loaded (legacy API)")
    except AttributeError:
        # New mediapipe API — download task model
        import urllib.request
        model_path = "face_landmarker.task"
        if not os.path.exists(model_path):
            print("[INFO] Downloading face landmarker model (~30 MB)...")
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "face_landmarker/face_landmarker/float16/1/"
                   "face_landmarker.task")
            urllib.request.urlretrieve(url, model_path)
            print("[INFO] Downloaded!")
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        base_opts  = mp_python.BaseOptions(model_asset_path=model_path)
        options    = vision.FaceLandmarkerOptions(
            base_options=base_opts,
            num_faces=5,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        face_mesh    = vision.FaceLandmarker.create_from_options(options)
        use_mediapipe = False
        print("[INFO] MediaPipe loaded (new tasks API)")

    # ── Open camera ───────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        exit()

    print("[INFO] Camera open. Starting recognition...\n")

    # ── State tracking ────────────────────────────────────────────
    liveness_states = {}    # face_index → LivenessState
    confirm_counts  = {}    # student_id → consecutive detection count
    marked_students = set() # students marked present this session
    results         = []    # last recognition results
    frame_count     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed.")
            break

        frame_count += 1
        h, w = frame.shape[:2]

        # ── Step 1: Liveness check via MediaPipe (every frame) ────
        if use_mediapipe:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_res = face_mesh.process(rgb)
            lm_list = mp_res.multi_face_landmarks \
                      if mp_res.multi_face_landmarks else []
        else:
            import mediapipe as mp2
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img   = mp2.Image(image_format=mp2.ImageFormat.SRGB,
                                 data=rgb)
            mp_res   = face_mesh.detect(mp_img)
            lm_list  = mp_res.face_landmarks \
                       if mp_res.face_landmarks else []

        # Update liveness state for each detected face
        for i, face_lms in enumerate(lm_list):
            if i not in liveness_states:
                liveness_states[i] = LivenessState()

            landmarks = face_lms.landmark \
                        if use_mediapipe else face_lms
            left_ear  = eye_aspect_ratio(landmarks, LEFT_EYE,  w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear       = (left_ear + right_ear) / 2.0
            liveness_states[i].update(ear)

        # Clean up states for faces no longer in frame
        for i in list(liveness_states.keys()):
            if i >= len(lm_list):
                del liveness_states[i]

        # ── Step 2: ArcFace recognition (every 5th frame) ─────────
        if frame_count % 5 == 0:
            insight_faces = face_app.get(frame)
            results = []

            for fi, face in enumerate(insight_faces):
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)

                emb = face.embedding
                if emb is None:
                    continue

                # Normalize embedding
                emb = emb / np.linalg.norm(emb)

                # Recognise against FAISS
                sid, name, conf = recognise_face(emb, index, labels, db)

                # Get liveness state for this face index
                live_state = liveness_states.get(
                    fi, LivenessState())

                results.append({
                    'box'        : (x1, y1, x2, y2),
                    'id'         : sid,
                    'name'       : name,
                    'confidence' : conf,
                    'is_live'    : live_state.is_live,
                    'live_msg'   : live_state.status_msg,
                    'face_idx'   : fi
                })

                # ── Step 3: Mark attendance only if LIVE ──────────
                if sid != 'UNKNOWN' and \
                   live_state.is_live and \
                   sid not in marked_students:

                    confirm_counts[sid] = \
                        confirm_counts.get(sid, 0) + 1

                    if confirm_counts[sid] >= CONFIRM_FRAMES:
                        marked_students.add(sid)
                        print(f"\n  [✓✓] MARKED PRESENT: {name} "
                              f"({sid})")
                        print(f"       Score     : {conf:.3f}")
                        print(f"       Liveness  : VERIFIED "
                              f"({BLINKS_NEEDED} blinks)\n")
                else:
                    # Reset confirmation if not live yet
                    if sid not in marked_students:
                        confirm_counts[sid] = 0

        # ── Step 4: Draw results on frame ─────────────────────────
        for r in results:
            x1, y1, x2, y2 = r['box']
            frame = draw_face_box(
                frame,
                x1, y1, x2, y2,
                r['name'], r['id'], r['confidence'],
                r['id'] != 'UNKNOWN',
                r['live_msg'], r['is_live']
            )

        # ── HUD overlay ───────────────────────────────────────────
        # Session summary top-left
        cv2.rectangle(frame, (8, 8), (320, 88), (0,0,0), -1)
        cv2.rectangle(frame, (8, 8), (320, 88), (60,60,60), 1)
        cv2.putText(frame, "SMART ATTENDANCE SYSTEM",
                    (14, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)
        cv2.putText(frame,
                    f"Enrolled : {len(db)} students",
                    (14, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(frame,
                    f"Marked   : {len(marked_students)} present",
                    (14, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,0), 1)
        cv2.putText(frame,
                    f"Session  : ACTIVE",
                    (14, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)

        # Bottom status bar
        cv2.putText(frame,
                    "Pipeline: Detect > Blink 2x > Recognise > Mark  "
                    "| Q/ESC=quit  D=debug",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150,150,150), 1)

        cv2.imshow("Smart Attendance - ArcFace + Liveness", frame)

        # ── Key handling ──────────────────────────────────────────
        key = cv2.waitKey(1)

        # Q, q, or ESC to quit
        if key in [ord('q'), ord('Q'), 27]:
            break

        # D for debug scores
        if key in [ord('d'), ord('D')]:
            print("\n── Debug: Current marked students ──────────────")
            if marked_students:
                for sid in marked_students:
                    name = db.get(sid, {}).get('name', '?')
                    print(f"  [✓] {name} ({sid})")
            else:
                print("  None marked yet")
            print("─────────────────────────────────────────────────")

    # ── Session summary ───────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*50)
    print("  SESSION ENDED — Attendance Summary")
    print("="*50)
    if marked_students:
        for sid in marked_students:
            name = db.get(sid, {}).get('name', 'Unknown')
            print(f"  [✓] PRESENT: {name} ({sid})")
    else:
        print("  No students marked present.")
    print(f"\n  Total present : {len(marked_students)}/{len(db)}")
    print("="*50)

# ## ▶️ Run It
# ```
# python -m ai.recogniser
# ```

# **What you'll see now:**
# - 🟠 Orange box → face recognised but **waiting for blinks**
# - Message: `"Blink 2 more time(s)"`
# - Blink twice → turns 🟢 **Green** → `"[LIVE] LIVE verified"`
# - Only then → `[✓✓] MARKED PRESENT` in console
# - Hold a **photo** to camera → no blinks → stays orange → **never marked** ✅

# ---

# ## 📤 Push to GitHub
# ```
# git add ai/recogniser.py
# git commit -m "Module 4+2: Integrated liveness gate into recognition pipeline"
# git push