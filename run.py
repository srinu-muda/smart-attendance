# run.py — Single entry point
# Starts Flask server + camera pipeline in one command:
#   python run.py

import os
import sys
import threading
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.makedirs("data", exist_ok=True)
os.makedirs("data/reports", exist_ok=True)

from flask import Flask, render_template, redirect, url_for
from flask_login import LoginManager, login_required
from app.models import db
from app.routes import api
from app.attendance import sync_students

# ── Absolute DB path ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "data", "attendance.db")

# ── Create Flask app ──────────────────────────────────────────────
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"]        = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"]                     = "smart_attendance_2026"

# ── Init DB + routes ──────────────────────────────────────────────
db.init_app(app)
app.register_blueprint(api)

# ── Pipeline thread state ─────────────────────────────────────────
pipeline_thread  = None
pipeline_running = False


def run_pipeline():
    """
    Run the attendance pipeline in a background thread.
    Imports here to avoid loading heavy models at startup.
    """
    global pipeline_running
    pipeline_running = True
    print("[PIPELINE] Starting in background thread...")

    try:
        import cv2
        import pickle
        import numpy as np
        import faiss
        import requests
        import mediapipe as mp
        from datetime import datetime
        from ai.encoder import get_face_app, load_db

        FAISS_INDEX          = "data/faiss.index"
        LABELS_FILE          = "data/labels.pkl"
        SIMILARITY_THRESHOLD = 0.68
        EAR_THRESHOLD        = 0.22
        CONSEC_FRAMES        = 2
        BLINKS_NEEDED        = 2
        CONFIRM_FRAMES       = 5
        LEFT_EYE  = [362, 385, 387, 263, 373, 380]
        RIGHT_EYE = [33,  160, 158, 133, 153, 144]

        def ear(lms, idx, w, h):
            pts = [(lms[i].x*w, lms[i].y*h) for i in idx]
            A = np.linalg.norm(np.array(pts[1])-np.array(pts[5]))
            B = np.linalg.norm(np.array(pts[2])-np.array(pts[4]))
            C = np.linalg.norm(np.array(pts[0])-np.array(pts[3]))
            return (A+B)/(2*C) if C else 0.0

        class LivenessState:
            def __init__(self):
                self.cnt     = 0
                self.blinks  = 0
                self.is_live = False
                self.msg     = f"Blink {BLINKS_NEEDED}x"
            def update(self, e):
                if e < EAR_THRESHOLD:
                    self.cnt += 1
                else:
                    if self.cnt >= CONSEC_FRAMES:
                        self.blinks += 1
                    self.cnt = 0
                if self.blinks >= BLINKS_NEEDED:
                    self.is_live = True
                    self.msg = "LIVE verified"
                else:
                    self.msg = f"Blink {BLINKS_NEEDED-self.blinks} more"

        # Load FAISS
        if not os.path.exists(FAISS_INDEX):
            print("[PIPELINE] No FAISS index. Enroll students first.")
            pipeline_running = False
            return

        index = faiss.read_index(FAISS_INDEX)
        with open(LABELS_FILE, "rb") as f:
            labels = pickle.load(f)
        face_db   = load_db()
        face_app  = get_face_app()

        # Load MediaPipe
        try:
            mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=10, refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
            legacy = True
        except:
            legacy = False
            mesh   = None

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[PIPELINE] Cannot open camera.")
            pipeline_running = False
            return

        print("[PIPELINE] Camera open. Waiting for active session...")

        liveness_states = {}
        confirm_counts  = {}
        marked_students = set()
        results         = []
        frame_count     = 0
        last_session_id = None

        while pipeline_running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame_count += 1
            h, w = frame.shape[:2]

            # Check active session from API
            if frame_count % 30 == 0:
                try:
                    r = requests.get(
                        "http://localhost:5000/api/session/active",
                        timeout=1)
                    s = r.json()
                    if s.get("status") == "ok":
                        sid = s["session"]["id"]
                        if sid != last_session_id:
                            last_session_id = sid
                            marked_students.clear()
                            confirm_counts.clear()
                            liveness_states.clear()
                            print(f"[PIPELINE] New session: "
                                  f"{s['session']['subject_code']}")
                    else:
                        last_session_id = None
                except:
                    pass

            # Only process if session active
            if not last_session_id:
                cv2.putText(frame,
                    "Waiting for session — start from dashboard",
                    (20, h//2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 165, 255), 2)
            else:
                # Liveness every frame
                if legacy and mesh:
                    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res    = mesh.process(rgb)
                    lm_list = res.multi_face_landmarks or []
                else:
                    lm_list = []

                for i, fl in enumerate(lm_list):
                    if i not in liveness_states:
                        liveness_states[i] = LivenessState()
                    lms = fl.landmark
                    e_val = (ear(lms,LEFT_EYE,w,h) +
                              ear(lms,RIGHT_EYE,w,h)) / 2
                    liveness_states[i].update(e_val)

                for i in list(liveness_states.keys()):
                    if i >= len(lm_list):
                        del liveness_states[i]

                # Recognition every 5th frame
                if frame_count % 5 == 0:
                    faces   = face_app.get(frame)
                    results = []
                    for fi, face in enumerate(faces):
                        x1,y1,x2,y2 = [int(v) for v in face.bbox]
                        x1=max(0,x1); y1=max(0,y1)
                        x2=min(w,x2); y2=min(h,y2)
                        emb = face.embedding
                        if emb is None: continue
                        emb = emb / np.linalg.norm(emb)
                        q   = emb.astype(np.float32).reshape(1,-1)
                        sc, idx = index.search(q, k=1)
                        best_sc  = float(sc[0][0])
                        best_idx = int(idx[0][0])
                        if best_sc >= SIMILARITY_THRESHOLD:
                            sid  = labels[best_idx]
                            name = face_db.get(sid,{}).get("name","?")
                        else:
                            sid  = "UNKNOWN"
                            name = "Unknown"
                        live = liveness_states.get(
                               fi, LivenessState())
                        results.append({
                            "box"  : (x1,y1,x2,y2),
                            "id"   : sid,
                            "name" : name,
                            "conf" : round(best_sc,3),
                            "live" : live.is_live,
                            "msg"  : live.msg
                        })
                        # Mark via API
                        if sid != "UNKNOWN" and \
                           live.is_live and \
                           sid not in marked_students:
                            confirm_counts[sid] = \
                                confirm_counts.get(sid,0) + 1
                            if confirm_counts[sid] >= CONFIRM_FRAMES:
                                try:
                                    requests.post(
                                        "http://localhost:5000/api/mark",
                                        json={"student_id": sid,
                                              "confidence": best_sc},
                                        timeout=1)
                                    marked_students.add(sid)
                                    print(f"[PIPELINE] MARKED: "
                                          f"{name} ({sid})")
                                except:
                                    pass
                        else:
                            if sid not in marked_students:
                                confirm_counts[sid] = 0

                # Draw results on frame
                for r in results:
                    x1,y1,x2,y2 = r["box"]
                    is_known = r["id"] != "UNKNOWN"
                    is_marked = r["id"] in marked_students
                    if is_marked:
                        col = (0,200,255)
                    elif is_known and r["live"]:
                        col = (0,210,0)
                    elif is_known:
                        col = (0,165,255)
                    else:
                        col = (0,0,220)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
                    label = f"{r['name']} {r['conf']:.0%}"
                    cv2.putText(frame, label, (x1,y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)
                    status = "[MARKED]" if is_marked else r["msg"]
                    cv2.putText(frame, status, (x1,y2+18),
                        cv2.FONT_HERSHEY_SIMPLEX,0.48,col,1)

            # HUD
            cv2.putText(frame,
                f"Session: {last_session_id or 'None'} | "
                f"Marked: {len(marked_students)}",
                (10,25), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0,255,255), 1)
            cv2.putText(frame,
                "Smart Attendance | Q=quit pipeline",
                (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.45,(150,150,150),1)

            cv2.imshow("Smart Attendance - Pipeline", frame)
            key = cv2.waitKey(1)
            if key in [ord("q"), ord("Q"), 27]:
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[PIPELINE] Stopped.")

    except Exception as e:
        print(f"[PIPELINE ERROR] {e}")

    pipeline_running = False


# ── Pipeline control API ──────────────────────────────────────────
@app.route("/api/pipeline/start", methods=["POST"])
def start_pipeline():
    global pipeline_thread, pipeline_running
    if pipeline_running:
        return {"status": "already_running"}
    pipeline_thread = threading.Thread(
        target=run_pipeline, daemon=True)
    pipeline_thread.start()
    return {"status": "ok", "msg": "Pipeline started"}


@app.route("/api/pipeline/stop", methods=["POST"])
def stop_pipeline():
    global pipeline_running
    pipeline_running = False
    return {"status": "ok", "msg": "Pipeline stopping..."}


@app.route("/api/pipeline/status", methods=["GET"])
def pipeline_status():
    return {"running": pipeline_running}


# ── Page routes ───────────────────────────────────────────────────
@app.route("/")
def dashboard():
    return render_template("dashboard.html")


# ── Startup ───────────────────────────────────────────────────────
with app.app_context():
    db.create_all()
    sync_students()
    print("[✓] Database ready")
    print("[✓] Students synced")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Smart AI Attendance System")
    print("  http://localhost:5000")
    print("  ONE terminal only — everything runs here")
    print("="*50 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000,
            threaded=True)


# ## ✅ Test the API

# Run the server:
# ```
# python run.py
# ```

# Open a **new terminal** and test each endpoint:
# ```
# curl http://localhost:5000/api/health
# ```
# ```
# curl http://localhost:5000/api/students
# ```
# ```
# curl -X POST http://localhost:5000/api/session/start -H "Content-Type: application/json" -d "{\"subject_code\":\"CS501\",\"subject_name\":\"AI\",\"faculty_name\":\"Prof. Test\"}"
# ```
# ```
# curl http://localhost:5000/api/session/active
# ```

# Or just open these URLs directly in your **browser**:
# ```
# http://localhost:5000/api/health
# http://localhost:5000/api/students
# http://localhost:5000/api/attendance/live
# ```

# ---

# ## 📤 Push to GitHub
# ```
# git add app/routes.py run.py
# git commit -m "Module 7: Flask REST API - all session and attendance endpoints"
# git push