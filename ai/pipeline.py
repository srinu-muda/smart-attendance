# ai/pipeline.py — CLEAN REWRITE with built-in debug
# Fixes: correct EAR threshold, stable liveness tracking

import cv2
import sys
import os
import pickle
import numpy as np
import faiss
import requests
import mediapipe as mp
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
from ai.encoder import get_face_app, load_db

# ══════════════════════════════════════════════════════════════════
# CONFIG — tuned for your EAR values (open=0.45, closed=0.03)
# ══════════════════════════════════════════════════════════════════
API_BASE             = "http://localhost:5000"
FAISS_INDEX          = "data/faiss.index"
LABELS_FILE          = "data/labels.pkl"
SIMILARITY_THRESHOLD = 0.65
EAR_THRESHOLD        = 0.28  # YOUR threshold from diagnostic
CONSEC_FRAMES        = 1      # 1 frame closed = blink
BLINKS_NEEDED        = 2      # 2 blinks to pass liveness
CONFIRM_FRAMES       = 5      # detections before marking

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

DEBUG = True   # Set False to hide debug prints


def log(msg):
    if DEBUG:
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ══════════════════════════════════════════════════════════════════
# EAR
# ══════════════════════════════════════════════════════════════════
def compute_ear(lms, indices, w, h):
    pts = [(lms[i].x * w, lms[i].y * h) for i in indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C) if C != 0 else 0.0


# ══════════════════════════════════════════════════════════════════
# LIVENESS TRACKER — simple and clean
# ══════════════════════════════════════════════════════════════════
class Liveness:
    def __init__(self):
        self.blink_counter = 0
        self.blinks        = 0
        self.is_live       = False
        self.last_ear      = 0.0

    def update(self, ear_val):
        self.last_ear = ear_val

        if ear_val < EAR_THRESHOLD:
            # Eye is closed this frame
            self.blink_counter += 1
        else:
            # Eye just opened
            if self.blink_counter >= CONSEC_FRAMES:
                # Completed one blink
                self.blinks += 1
                log(f"BLINK! count={self.blinks}/{BLINKS_NEEDED} "
                    f"ear_was={ear_val:.3f}")
            self.blink_counter = 0

        if self.blinks >= BLINKS_NEEDED:
            self.is_live = True

    @property
    def msg(self):
        if self.is_live:
            return "LIVE VERIFIED"
        return f"Blink {BLINKS_NEEDED - self.blinks} more"


# ══════════════════════════════════════════════════════════════════
# FAISS + API
# ══════════════════════════════════════════════════════════════════
def load_faiss():
    if not os.path.exists(FAISS_INDEX):
        print("[ERROR] No FAISS index. Enroll students first.")
        return None, None
    index = faiss.read_index(FAISS_INDEX)
    with open(LABELS_FILE, "rb") as f:
        labels = pickle.load(f)
    return index, labels


def api_get(endpoint):
    try:
        return requests.get(
            f"{API_BASE}{endpoint}", timeout=2).json()
    except:
        return {"status": "error"}


def api_post(endpoint, data):
    try:
        return requests.post(
            f"{API_BASE}{endpoint}",
            json=data, timeout=2).json()
    except:
        return {"status": "error"}


# ══════════════════════════════════════════════════════════════════
# DRAW
# ══════════════════════════════════════════════════════════════════
def draw_box(frame, x1, y1, x2, y2,
             name, sid, conf, liveness, is_marked):
    is_known = sid != "UNKNOWN"

    if is_marked:
        color = (0, 200, 255)
    elif liveness.is_live and is_known:
        color = (0, 210, 0)
    elif is_known:
        color = (0, 140, 255)
    else:
        color = (0, 0, 200)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Name label
    label = f"{name} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame,
                  (x1, y1 - th - 12),
                  (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

    # Liveness status
    live_text = "[MARKED]" if is_marked else liveness.msg
    cv2.putText(frame, live_text,
                (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1)

    # EAR + blink count
    cv2.putText(frame,
                f"EAR:{liveness.last_ear:.3f} "
                f"Blinks:{liveness.blinks}/{BLINKS_NEEDED}",
                (x1, y2 + 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (200, 200, 0), 1)

    # Blink progress bar
    bw   = x2 - x1
    prog = int(bw * min(liveness.blinks / BLINKS_NEEDED, 1.0))
    cv2.rectangle(frame, (x1, y2+40), (x2, y2+46),
                  (50, 50, 50), -1)
    cv2.rectangle(frame, (x1, y2+40),
                  (x1 + prog, y2+46), color, -1)
    return frame


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  Smart Attendance Pipeline — Clean Rewrite")
    print(f"  EAR threshold : {EAR_THRESHOLD}")
    print(f"  Blinks needed : {BLINKS_NEEDED}")
    print(f"  Confirm frames: {CONFIRM_FRAMES}")
    print("=" * 55)

    # ── Flask check ───────────────────────────────────────────────
    print("\n[1/5] Checking Flask server...")
    health = api_get("/api/health")
    if health.get("status") != "ok":
        print("[ERROR] Flask not running! Run: python run.py")
        exit()
    print("[OK] Flask running")

    # ── Wait for session ──────────────────────────────────────────
    print("[2/5] Checking active session...")
    sess = api_get("/api/session/active")
    if sess.get("status") != "ok":
        print("[WAIT] No session. Start one from dashboard:")
        print("       http://localhost:5000")
        while True:
            try:
                time.sleep(3)
                sess = api_get("/api/session/active")
                if sess.get("status") == "ok":
                    print("[OK] Session found!")
                    break
                print("      Still waiting...")
            except KeyboardInterrupt:
                exit()

    session_info = sess["session"]
    session_id   = session_info["id"]
    print(f"[OK] Session: {session_info['subject_code']} "
          f"(ID:{session_id})")

    # ── Load FAISS ────────────────────────────────────────────────
    print("[3/5] Loading FAISS index...")
    index, labels = load_faiss()
    if index is None:
        exit()
    face_db        = load_db()
    total_students = len(face_db)
    print(f"[OK] {index.ntotal} vectors, "
          f"{total_students} students")

    # ── Load InsightFace ──────────────────────────────────────────
    print("[4/5] Loading InsightFace...")
    face_app = get_face_app()
    print("[OK] InsightFace ready")

    # ── Load MediaPipe ────────────────────────────────────────────
    print("[5/5] Loading MediaPipe...")
    try:
        face_mesh  = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3)
        use_legacy = True
        print("[OK] MediaPipe legacy API")
    except AttributeError:
        import urllib.request
        model_path = "face_landmarker.task"
        if not os.path.exists(model_path):
            print("[INFO] Downloading model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models"
                "/face_landmarker/face_landmarker/float16/1"
                "/face_landmarker.task", model_path)
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        face_mesh = vision.FaceLandmarker\
                        .create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=model_path),
                num_faces=5,
                min_face_detection_confidence=0.3,
                min_tracking_confidence=0.3))
        use_legacy = False
        print("[OK] MediaPipe new API")

    # ── Camera ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera failed!")
        exit()

    print("\n[READY] Pipeline running!")
    print(f"        BLINK {BLINKS_NEEDED} TIMES to mark present")
    print("        Q = quit\n")

    # ── State ─────────────────────────────────────────────────────
    # student_id → Liveness object
    # We track per STUDENT not per face index
    # so state survives frame-to-frame
    liveness_map    = {}   # student_id → Liveness
    unknown_live    = Liveness()  # for unknown faces
    confirm_counts  = {}   # student_id → int
    marked_students = set()
    results         = []
    frame_count     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w = frame.shape[:2]

        # ── MediaPipe landmarks (every frame) ─────────────────────
        if use_legacy:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_res  = face_mesh.process(rgb)
            lm_list = mp_res.multi_face_landmarks or []
        else:
            import mediapipe as mp2
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img  = mp2.Image(
                image_format=mp2.ImageFormat.SRGB, data=rgb)
            mp_res  = face_mesh.detect(mp_img)
            lm_list = mp_res.face_landmarks or []

        # ── InsightFace recognition (every 4th frame) ─────────────
        if frame_count % 4 == 0:
            faces   = face_app.get(frame)
            results = []

            for fi, face in enumerate(faces):
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)

                # ── Recognise ─────────────────────────────────────
                emb = face.embedding
                if emb is None:
                    continue
                emb   = emb / np.linalg.norm(emb)
                emb_q = emb.astype(np.float32).reshape(1, -1)
                sc, idx = index.search(emb_q, k=1)
                best_sc  = float(sc[0][0])
                best_idx = int(idx[0][0])

                if best_sc >= SIMILARITY_THRESHOLD \
                   and best_idx >= 0:
                    sid  = labels[best_idx]
                    name = face_db.get(
                        sid, {}).get("name", "?")
                else:
                    sid  = "UNKNOWN"
                    name = "Unknown"

                # ── Get EAR from matching MediaPipe face ──────────
                ear_val = 0.0
                if fi < len(lm_list):
                    lms = lm_list[fi].landmark \
                          if use_legacy else lm_list[fi]
                    l_e = compute_ear(lms, LEFT_EYE,  w, h)
                    r_e = compute_ear(lms, RIGHT_EYE, w, h)
                    ear_val = (l_e + r_e) / 2.0

                # ── Update liveness tracked per student_id ────────
                if sid != "UNKNOWN":
                    if sid not in liveness_map:
                        liveness_map[sid] = Liveness()
                    live = liveness_map[sid]
                    live.update(ear_val)   # Only update if known
                else:
                    live = unknown_live    # Don't update unknown

                live.update(ear_val)

                results.append({
                    "box" : (x1, y1, x2, y2),
                    "sid" : sid,
                    "name": name,
                    "conf": round(best_sc, 3),
                    "live": live
                })

                # ── Debug every 20 frames ─────────────────────────
                if frame_count % 20 == 0 and DEBUG:
                    log(f"sid={sid:12s} "
                        f"ear={ear_val:.3f} "
                        f"blinks={live.blinks} "
                        f"live={live.is_live} "
                        f"conf_count="
                        f"{confirm_counts.get(sid,0)} "
                        f"marked="
                        f"{sid in marked_students}")

                # ── Mark present if conditions met ─────────────────
                if sid != "UNKNOWN" and \
                   live.is_live and \
                   sid not in marked_students:

                    confirm_counts[sid] = \
                        confirm_counts.get(sid, 0) + 1

                    log(f"Confirming {name}: "
                        f"{confirm_counts[sid]}/{CONFIRM_FRAMES}")

                    if confirm_counts[sid] >= CONFIRM_FRAMES:
                        res = api_post("/api/mark", {
                            "student_id": sid,
                            "confidence": best_sc
                        })
                        if res.get("status") in \
                           ["ok", "duplicate"]:
                            marked_students.add(sid)
                            print(f"\n  [✓✓] MARKED: {name}"
                                  f" ({sid})"
                                  f" score={best_sc:.3f}\n")
                        else:
                            print(f"  [API ERR] {res}")

        # ── Draw ──────────────────────────────────────────────────
        for r in results:
            x1, y1, x2, y2 = r["box"]
            frame = draw_box(
                frame, x1, y1, x2, y2,
                r["name"], r["sid"], r["conf"],
                r["live"],
                r["sid"] in marked_students)

        # HUD
        cv2.rectangle(frame, (6, 6), (300, 70),
                      (0, 0, 0), -1)
        cv2.putText(frame,
                    f"Session: "
                    f"{session_info.get('subject_code')}",
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (0, 255, 255), 1)
        cv2.putText(frame,
                    f"Marked: {len(marked_students)}"
                    f"/{total_students}  "
                    f"EAR_thr={EAR_THRESHOLD}",
                    (12, 46),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0, 220, 0), 1)
        cv2.putText(frame,
                    f"Q=quit  "
                    f"Blink {BLINKS_NEEDED}x to mark",
                    (12, 64),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (150, 150, 150), 1)

        cv2.imshow("Smart Attendance", frame)

        key = cv2.waitKey(1)
        if key in [ord("q"), ord("Q"), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "="*50)
    print("  PIPELINE ENDED")
    print("="*50)
    for sid in marked_students:
        n = face_db.get(sid, {}).get("name", "?")
        print(f"  [✓] {n} ({sid})")
    if not marked_students:
        print("  Nobody marked.")
    print(f"  Total: {len(marked_students)}/{total_students}")
    print("="*50)

# ## ▶️ Run It

# **Terminal 1:**
# ```
# python run.py
# ```

# **Browser:** `http://localhost:5000` → Start Session

# **Terminal 2:**
# ```
# python -m ai.pipeline
# ```

# Watch the terminal — you'll see debug lines every 20 frames like:
# ```
# [15:42:10] sid=U24CS167   ear=0.031 blinks=0 live=False conf_count=0 marked=False
# [15:42:11] BLINK! count=1/2 ear_was=0.031
# [15:42:12] BLINK! count=2/2 ear_was=0.028
# [15:42:12] sid=U24CS167   ear=0.420 blinks=2 live=True conf_count=1 marked=False
# [15:42:13] Confirming Maneesh: 5/5
# [✓✓] MARKED: DASARI MANEESH KUMAR (U24CS167) score=0.891