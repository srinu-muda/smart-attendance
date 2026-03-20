# ai/pipeline.py
# Module 8: Complete Live Attendance Pipeline
# Flow: Camera → Detect → Liveness → Recognise → API mark present
# Run this during actual class to auto-mark attendance

import cv2
import sys
import os
import pickle
import numpy as np
import faiss
import requests
import mediapipe as mp
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.encoder import get_face_app, load_db

# ── Config ────────────────────────────────────────────────────────
API_BASE         = "http://localhost:5000"   # Flask server URL
FAISS_INDEX      = "data/faiss.index"
LABELS_FILE      = "data/labels.pkl"
SIMILARITY_THRESHOLD = 0.68
EAR_THRESHOLD        = 0.22
CONSEC_FRAMES        = 2
BLINKS_NEEDED        = 2
CONFIRM_FRAMES       = 5

# ── Eye landmark indices (MediaPipe) ─────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def eye_aspect_ratio(landmarks, indices, w, h):
    """Compute EAR for one eye."""
    pts = [(landmarks[i].x * w,
            landmarks[i].y * h) for i in indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C) if C != 0 else 0.0


def load_faiss():
    """Load FAISS index and labels."""
    if not os.path.exists(FAISS_INDEX) or \
       not os.path.exists(LABELS_FILE):
        print("[ERROR] FAISS index not found.")
        print("[ERROR] Run: python -m ai.encoder first.")
        return None, None
    index = faiss.read_index(FAISS_INDEX)
    with open(LABELS_FILE, "rb") as f:
        labels = pickle.load(f)
    return index, labels


def api_mark_present(student_id, confidence):
    """
    Call Flask API to mark student present.
    Returns server response dict.
    """
    try:
        res = requests.post(
            f"{API_BASE}/api/mark",
            json={"student_id": student_id,
                  "confidence": confidence},
            timeout=2
        )
        return res.json()
    except requests.exceptions.ConnectionError:
        return {"status": "error",
                "msg"   : "Flask server not running!"}
    except Exception as e:
        return {"status": "error", "msg": str(e)}


def api_get_active_session():
    """Check if a session is active on the server."""
    try:
        res = requests.get(f"{API_BASE}/api/session/active",
                           timeout=2)
        return res.json()
    except:
        return {"status": "error"}


def draw_hud(frame, session_info, marked, total,
             pipeline_active):
    """
    Draw info overlay on top-left of frame.
    Shows session info, marked count, pipeline status.
    """
    h, w = frame.shape[:2]

    # Dark background panel
    cv2.rectangle(frame, (8, 8), (340, 110), (0, 0, 0), -1)
    cv2.rectangle(frame, (8, 8), (340, 110), (60, 60, 60), 1)

    # Title
    cv2.putText(frame, "SMART ATTENDANCE SYSTEM",
                (14, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 1)

    # Session info
    subj = session_info.get("subject_code", "No session")
    cv2.putText(frame, f"Subject : {subj}",
                (14, 47), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)

    # Marked count
    cv2.putText(frame,
                f"Marked  : {len(marked)}/{total} present",
                (14, 65), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 220, 0), 1)

    # Time
    cv2.putText(frame,
                f"Time    : {datetime.now().strftime('%H:%M:%S')}",
                (14, 83), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)

    # Pipeline status
    status_color = (0, 220, 0) if pipeline_active else (0, 0, 220)
    status_text  = "ACTIVE" if pipeline_active else "PAUSED"
    cv2.putText(frame, f"Status  : {status_text}",
                (14, 101), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, status_color, 1)

    # Bottom help bar
    cv2.putText(frame,
                "Q=quit  P=pause  D=debug  "
                "S=session summary",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (150, 150, 150), 1)
    return frame


def draw_face_result(frame, x1, y1, x2, y2,
                     name, sid, confidence,
                     is_live, live_msg, is_marked):
    """Draw face box with recognition + liveness status."""
    # Color logic
    if is_marked:
        color = (0, 200, 255)    # Cyan = already marked
    elif is_live and sid != "UNKNOWN":
        color = (0, 210, 0)      # Green = live + recognised
    elif sid != "UNKNOWN":
        color = (0, 165, 255)    # Orange = recognised, not live yet
    else:
        color = (0, 0, 220)      # Red = unknown

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Name label background
    label = f"{name}  {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    cv2.rectangle(frame,
                  (x1, y1 - th - 14),
                  (x1 + tw + 10, y1),
                  color, -1)
    cv2.putText(frame, label,
                (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (255, 255, 255), 2)

    # Student ID
    if sid != "UNKNOWN":
        cv2.putText(frame, f"ID: {sid}",
                    (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0, 255, 255), 1)

    # Liveness / marked status
    if is_marked:
        live_color = (0, 200, 255)
        live_text  = "[MARKED PRESENT]"
    else:
        live_color = (0, 210, 0) if is_live else (0, 165, 255)
        live_text  = f"{'[LIVE]' if is_live else '[?]'} {live_msg}"

    cv2.putText(frame, live_text,
                (x1, y2 + 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48, live_color, 1)

    # Confidence bar
    bw   = x2 - x1
    fill = int(bw * min(confidence, 1.0))
    cv2.rectangle(frame, (x1, y2+42), (x2, y2+48),
                  (50, 50, 50), -1)
    cv2.rectangle(frame, (x1, y2+42),
                  (x1 + fill, y2+48), color, -1)

    return frame


# ══════════════════════════════════════════════════════════════════
# LIVENESS STATE TRACKER
# ══════════════════════════════════════════════════════════════════

class LivenessState:
    def __init__(self):
        self.blink_counter = 0
        self.total_blinks  = 0
        self.is_live       = False
        self.status_msg    = f"Blink {BLINKS_NEEDED}x to verify"

    def update(self, ear):
        if ear < EAR_THRESHOLD:
            self.blink_counter += 1
        else:
            if self.blink_counter >= CONSEC_FRAMES:
                self.total_blinks += 1
            self.blink_counter = 0

        remaining = max(0, BLINKS_NEEDED - self.total_blinks)
        if self.total_blinks >= BLINKS_NEEDED:
            self.is_live    = True
            self.status_msg = "LIVE verified"
        else:
            self.status_msg = f"Blink {remaining} more time(s)"


# ══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 58)
    print("  SMART ATTENDANCE — Full Pipeline")
    print("  Detect → Liveness → Recognise → Mark")
    print("=" * 58)

    # ── Check Flask server is running ─────────────────────────────
    print("\n[INFO] Checking Flask server...")
    session_res = api_get_active_session()
    if session_res.get("status") == "error":
        print("[ERROR] Flask server not running!")
        print("[ERROR] Open a new terminal and run:")
        print("        python run.py")
        print("[ERROR] Then run this pipeline again.")
        exit()
    print("[INFO] Flask server OK")

    # ── Check active session ──────────────────────────────────────
    if session_res.get("status") == "none":
        print("\n[WARN] No active session found.")
        print("Start one? (y/n): ", end="")
        if input().strip().lower() == "y":
            code = input("Subject code (e.g. CS501): ").strip()
            name = input("Subject name             : ").strip()
            try:
                res = requests.post(
                    f"{API_BASE}/api/session/start",
                    json={"subject_code": code,
                          "subject_name": name},
                    timeout=2
                )
                session_res = res.json()
                print(f"[✓] Session started: {code}")
            except:
                print("[ERROR] Could not start session.")
                exit()
        else:
            print("[INFO] Exiting — start a session first.")
            exit()

    session_info = session_res.get("session", {})
    print(f"\n[INFO] Active session: "
          f"{session_info.get('subject_code')} "
          f"(ID: {session_info.get('id')})\n")

    # ── Load FAISS + face DB ──────────────────────────────────────
    index, labels = load_faiss()
    if index is None:
        exit()
    db = load_db()
    total_students = len(db)
    print(f"[INFO] {total_students} student(s) in face DB")

    # ── Load InsightFace ──────────────────────────────────────────
    face_app = get_face_app()

    # ── Load MediaPipe ────────────────────────────────────────────
    print("[INFO] Loading MediaPipe liveness detector...")
    try:
        mp_mesh   = mp.solutions.face_mesh
        face_mesh = mp_mesh.FaceMesh(
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        use_legacy = True
    except AttributeError:
        import urllib.request
        model_path = "face_landmarker.task"
        if not os.path.exists(model_path):
            print("[INFO] Downloading liveness model...")
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "face_landmarker/face_landmarker/float16/1/"
                   "face_landmarker.task")
            urllib.request.urlretrieve(url, model_path)
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        opts      = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=model_path),
            num_faces=10,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        face_mesh  = vision.FaceLandmarker.create_from_options(opts)
        use_legacy = False
    print("[INFO] MediaPipe loaded")

    # ── Open camera ───────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        exit()

    print("[INFO] Camera open. Pipeline running...\n")

    # ── State ─────────────────────────────────────────────────────
    liveness_states = {}   # face_idx → LivenessState
    confirm_counts  = {}   # student_id → int
    marked_students = set()
    results         = []
    frame_count     = 0
    pipeline_active = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w = frame.shape[:2]

        if not pipeline_active:
            cv2.putText(frame, "PIPELINE PAUSED — Press P to resume",
                        (w//2 - 200, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
        else:
            # ── Liveness (every frame) ────────────────────────────
            if use_legacy:
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_res = face_mesh.process(rgb)
                lm_list = mp_res.multi_face_landmarks or []
            else:
                import mediapipe as mp2
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img  = mp2.Image(
                    image_format=mp2.ImageFormat.SRGB, data=rgb)
                mp_res  = face_mesh.detect(mp_img)
                lm_list = mp_res.face_landmarks or []

            for i, face_lms in enumerate(lm_list):
                if i not in liveness_states:
                    liveness_states[i] = LivenessState()
                lms  = face_lms.landmark \
                       if use_legacy else face_lms
                ear  = (eye_aspect_ratio(lms, LEFT_EYE, w, h) +
                        eye_aspect_ratio(lms, RIGHT_EYE, w, h)) / 2
                liveness_states[i].update(ear)

            # Clean stale liveness states
            for i in list(liveness_states.keys()):
                if i >= len(lm_list):
                    del liveness_states[i]

            # ── Recognition (every 5th frame) ─────────────────────
            if frame_count % 5 == 0:
                insight_faces = face_app.get(frame)
                results       = []

                for fi, face in enumerate(insight_faces):
                    x1,y1,x2,y2 = [int(v) for v in face.bbox]
                    x1 = max(0,x1); y1 = max(0,y1)
                    x2 = min(w,x2); y2 = min(h,y2)

                    emb = face.embedding
                    if emb is None:
                        continue
                    emb = emb / np.linalg.norm(emb)
                    emb_q = emb.astype(np.float32).reshape(1,-1)

                    scores, indices = index.search(emb_q, k=1)
                    best_score = float(scores[0][0])
                    best_idx   = int(indices[0][0])

                    if best_score >= SIMILARITY_THRESHOLD \
                       and best_idx >= 0:
                        sid  = labels[best_idx]
                        name = db.get(sid, {}).get("name", "?")
                    else:
                        sid  = "UNKNOWN"
                        name = "Unknown"

                    live = liveness_states.get(fi, LivenessState())

                    results.append({
                        "box"       : (x1,y1,x2,y2),
                        "id"        : sid,
                        "name"      : name,
                        "confidence": round(best_score, 3),
                        "is_live"   : live.is_live,
                        "live_msg"  : live.status_msg,
                        "face_idx"  : fi
                    })

                    # ── Mark via API if live + confirmed ──────────
                    if sid != "UNKNOWN" and \
                       live.is_live and \
                       sid not in marked_students:

                        confirm_counts[sid] = \
                            confirm_counts.get(sid, 0) + 1

                        if confirm_counts[sid] >= CONFIRM_FRAMES:
                            api_res = api_mark_present(
                                sid, best_score)

                            if api_res.get("status") == "ok":
                                marked_students.add(sid)
                                print(f"\n  [✓✓] MARKED: {name} "
                                      f"({sid})  "
                                      f"score={best_score:.3f}")
                                print(f"        Liveness: VERIFIED")
                                print(f"        Time    : "
                                      f"{datetime.now().strftime('%H:%M:%S')}\n")
                            elif api_res.get("status") == \
                                 "duplicate":
                                marked_students.add(sid)
                    else:
                        if sid not in marked_students:
                            confirm_counts[sid] = 0

        # ── Draw everything ───────────────────────────────────────
        for r in results:
            x1,y1,x2,y2 = r["box"]
            frame = draw_face_result(
                frame, x1,y1,x2,y2,
                r["name"], r["id"], r["confidence"],
                r["is_live"], r["live_msg"],
                r["id"] in marked_students
            )

        frame = draw_hud(frame, session_info,
                         marked_students,
                         total_students,
                         pipeline_active)

        cv2.imshow("Smart Attendance — Live Pipeline", frame)

        # ── Key handling ──────────────────────────────────────────
        key = cv2.waitKey(1)

        if key in [ord("q"), ord("Q"), 27]:   # Quit
            break

        if key in [ord("p"), ord("P")]:       # Pause/resume
            pipeline_active = not pipeline_active
            state = "PAUSED" if not pipeline_active else "RESUMED"
            print(f"[INFO] Pipeline {state}")

        if key in [ord("s"), ord("S")]:       # Session summary
            print(f"\n── Session Summary ─────────────────────────")
            print(f"  Subject : "
                  f"{session_info.get('subject_code')}")
            print(f"  Marked  : {len(marked_students)}"
                  f"/{total_students}")
            for sid in marked_students:
                n = db.get(sid, {}).get("name", "?")
                print(f"  [✓] {n} ({sid})")
            print(f"────────────────────────────────────────────\n")

    # ── Cleanup ───────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*50)
    print("  PIPELINE ENDED — Final Summary")
    print("="*50)
    for sid in marked_students:
        name = db.get(sid, {}).get("name", "Unknown")
        print(f"  [✓] PRESENT: {name} ({sid})")
    if not marked_students:
        print("  No students marked present.")
    print(f"\n  Total: {len(marked_students)}/{total_students}")
    print("="*50)

# ## ▶️ How to Run — Two Terminals Needed

# **Terminal 1 — Start Flask server:**
# ```
# python run.py
# ```

# **Terminal 2 — Start pipeline:**
# ```
# python -m ai.pipeline
# ```

# The pipeline will ask if you want to start a session — type `y`, enter subject code and name.

# ---

# ## 📤 Push to GitHub
# ```
# git add ai/pipeline.py
# git commit -m "Module 8: Complete live attendance pipeline - detect+liveness+recognise+mark"
# git push