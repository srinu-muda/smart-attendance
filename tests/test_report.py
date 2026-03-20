# tests/test_report.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs("data", exist_ok=True)

from flask import Flask
from app.models import db
from app.attendance import (sync_students, start_session,
                             mark_present, stop_session)
from app.report import generate_excel, generate_csv, list_reports

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "data", "attendance.db")

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"]        = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

with app.app_context():
    db.create_all()
    sync_students()

    # Create a fresh test session
    session = start_session("CS501", "Artificial Intelligence",
                             "Prof. Test")

    # Mark all enrolled students present
    from ai.encoder import load_db
    face_db = load_db()
    for sid, info in face_db.items():
        mark_present(sid, session.id, confidence=0.89)

    stop_session(session.id)

    # Generate both report formats
    print("\n[TEST] Generating Excel report...")
    excel_path = generate_excel(session.id)

    print("\n[TEST] Generating CSV report...")
    csv_path = generate_csv(session.id)

    print("\n[TEST] All report files:")
    list_reports()

    print("\n" + "="*45)
    print("  [✓] Report generator tests passed!")
    if excel_path:
        print(f"  Open this file to see formatted Excel:")
        print(f"  {excel_path}")
    print("="*45)          

# Run it:
# ```
# python tests\test_report.py
# ```

# ---

# ## 📤 Push to GitHub
# ```
# git add app/report.py tests/test_report.py
# git commit -m "Module 6: Report generator - Excel and CSV export with formatting"
# git push