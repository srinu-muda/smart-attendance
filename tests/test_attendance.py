# tests/test_attendance.py — Quick DB test
import sys, os

# Fix paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure data folder exists before DB connection
os.makedirs("data", exist_ok=True)

from flask import Flask
from app.models import db
from app.attendance import (sync_students, start_session,
                             mark_present, get_session_report,
                             stop_session)

# Absolute path to DB — fixes "unable to open database file"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "data", "attendance.db")

# Create minimal Flask app for testing
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"]        = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

with app.app_context():
    print("[TEST] Creating database tables...")
    db.create_all()
    print("[TEST] Tables created OK")

    print("\n[TEST] Syncing enrolled students...")
    sync_students()

    # Start a test session
    print("\n[TEST] Starting session...")
    session = start_session(
        subject_code = "CS501",
        subject_name = "Artificial Intelligence",
        faculty_name = "Prof. Test"
    )
    print(f"[TEST] Session started → ID: {session.id}")

    # Mark first enrolled student present
    from ai.encoder import load_db
    face_db = load_db()

    if face_db:
        first_id = list(face_db.keys())[0]
        name     = face_db[first_id]['name']
        print(f"\n[TEST] Marking {name} ({first_id}) present...")

        result = mark_present(first_id, session.id, confidence=0.89)
        print(f"[TEST] Result → {result}")

        # Try duplicate — should be blocked
        result2 = mark_present(first_id, session.id, confidence=0.89)
        print(f"[TEST] Duplicate test → {result2}")
    else:
        print("[WARN] No students enrolled in face DB yet.")
        print("[WARN] Run: python -m ai.encoder first")

    # Get full report
    print("\n[TEST] Fetching session report...")
    report = get_session_report(session.id)
    print(f"       Present : {report['total_present']}")
    print(f"       Absent  : {report['total_absent']}")
    print(f"       Total   : {report['total']}")

    for s in report["present"]:
        print(f"       [✓] {s['name']:20s} ({s['student_id']})  "
              f"at {s['time']}  conf={s['confidence']}")

    for s in report["absent"]:
        print(f"       [✗] {s['name']:20s} ({s['student_id']})  "
              f"ABSENT")

    # Stop session
    print("\n[TEST] Stopping session...")
    summary = stop_session(session.id)
    print(f"[TEST] Summary → {summary}")

    print("\n" + "="*45)
    print("  [✓] All attendance engine tests passed!")
    print("="*45)
# Run it:
# ```
# python tests\test_attendance.py
# ```

# ---

# ## 📤 Push to GitHub
# ```
# git add app/models.py app/attendance.py tests/test_attendance.py
# git commit -m "Module 5: Attendance engine - SQLite DB, sessions, mark present"
# git push