# run.py — Main Flask application entry point
# Run this file to start the complete Smart Attendance System

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.makedirs("data", exist_ok=True)
os.makedirs("data/reports", exist_ok=True)

from flask import Flask, render_template
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

# ── Init extensions ───────────────────────────────────────────────
db.init_app(app)
app.register_blueprint(api)

# ── Dashboard route ───────────────────────────────────────────────
@app.route("/")
def dashboard():
    """Serve the main dashboard page."""
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
    print("  Press CTRL+C to stop")
    print("="*50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)


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