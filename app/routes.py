# app/routes.py
# Module 7: Flask REST API
# All endpoints for session management, attendance, reports

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint, jsonify, request, send_file
from app.models import db, Student, Session, AttendanceLog
from app.attendance import (sync_students, start_session,
                             stop_session, mark_present,
                             get_session_report,
                             get_active_session)
from app.report import generate_excel, generate_csv
# Add these imports at top of routes.py
from flask import session as flask_session, render_template
from functools import wraps
from app.models import Faculty
from flask_bcrypt import Bcrypt

bcrypt = Bcrypt()

def login_required_custom(f):
    """Simple login check decorator."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not flask_session.get("faculty_id"):
            from flask import redirect
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated


# ── Blueprint for all routes ──────────────────────────────────────
api = Blueprint("api", __name__)


# ══════════════════════════════════════════════════════════════════
# STUDENT ROUTES
# ══════════════════════════════════════════════════════════════════

@api.route("/api/students", methods=["GET"])
def get_students():
    """Get all enrolled students."""
    sync_students()   # Sync face DB → Student table
    students = Student.query.all()
    return jsonify({
        "status"  : "ok",
        "count"   : len(students),
        "students": [s.to_dict() for s in students]
    })


# ══════════════════════════════════════════════════════════════════
# SESSION ROUTES
# ══════════════════════════════════════════════════════════════════

@api.route("/api/session/start", methods=["POST"])
def api_start_session():
    """
    Start a new attendance session.

    Body (JSON):
        subject_code : e.g. "CS501"
        subject_name : e.g. "Artificial Intelligence"
        faculty_name : e.g. "Prof. Sharma"
    """
    data         = request.get_json() or {}
    subject_code = data.get("subject_code", "UNKNOWN")
    subject_name = data.get("subject_name", "")
    faculty_name = data.get("faculty_name", "")

    if not subject_code:
        return jsonify({"status": "error",
                        "msg"   : "subject_code is required"}), 400

    session = start_session(subject_code,
                             subject_name,
                             faculty_name)
    return jsonify({
        "status" : "ok",
        "msg"    : f"Session started for {subject_code}",
        "session": session.to_dict()
    })


@api.route("/api/session/stop", methods=["POST"])
def api_stop_session():
    """Stop the currently active session."""
    session = get_active_session()
    if not session:
        return jsonify({"status": "error",
                        "msg"   : "No active session found"}), 404

    summary = stop_session(session.id)
    return jsonify({
        "status" : "ok",
        "msg"    : "Session stopped",
        "summary": summary
    })


@api.route("/api/session/active", methods=["GET"])
def api_active_session():
    """Get currently active session."""
    session = get_active_session()
    if not session:
        return jsonify({"status": "none",
                        "msg"   : "No active session"})
    return jsonify({
        "status" : "ok",
        "session": session.to_dict()
    })


@api.route("/api/sessions", methods=["GET"])
def api_all_sessions():
    """Get all past sessions."""
    sessions = Session.query.order_by(
                   Session.start_time.desc()).all()
    return jsonify({
        "status"  : "ok",
        "count"   : len(sessions),
        "sessions": [s.to_dict() for s in sessions]
    })


# ══════════════════════════════════════════════════════════════════
# ATTENDANCE ROUTES
# ══════════════════════════════════════════════════════════════════

@api.route("/api/mark", methods=["POST"])
def api_mark_present():
    """
    Mark a student present in the active session.

    Body (JSON):
        student_id : e.g. "U24CS167"
        confidence : float 0.0–1.0
    """
    data       = request.get_json() or {}
    student_id = data.get("student_id")
    confidence = data.get("confidence", 1.0)

    if not student_id:
        return jsonify({"status": "error",
                        "msg"   : "student_id is required"}), 400

    # Get active session
    session = get_active_session()
    if not session:
        return jsonify({"status": "error",
                        "msg"   : "No active session. "
                                  "Start a session first."}), 400

    result = mark_present(student_id,
                           session.id,
                           confidence)
    return jsonify(result)


@api.route("/api/attendance/<int:session_id>", methods=["GET"])
def api_get_attendance(session_id):
    """Get full attendance report for a session."""
    report = get_session_report(session_id)
    if "error" in report:
        return jsonify({"status": "error",
                        "msg"   : report["error"]}), 404
    return jsonify({"status": "ok", **report})


@api.route("/api/attendance/live", methods=["GET"])
def api_live_attendance():
    """
    Get live attendance for the active session.
    Called by dashboard to update table in real-time.
    """
    session = get_active_session()
    if not session:
        return jsonify({"status": "none",
                        "present": [],
                        "total_present": 0})

    report = get_session_report(session.id)
    return jsonify({
        "status"       : "ok",
        "session_id"   : session.id,
        "subject"      : session.subject_code,
        "present"      : report["present"],
        "absent"       : report["absent"],
        "total_present": report["total_present"],
        "total_absent" : report["total_absent"],
        "total"        : report["total"]
    })


# ══════════════════════════════════════════════════════════════════
# REPORT ROUTES
# ══════════════════════════════════════════════════════════════════

@api.route("/api/report/<int:session_id>/excel", methods=["GET"])
def api_export_excel(session_id):
    """Download Excel attendance report for a session."""
    filepath = generate_excel(session_id)
    if not filepath:
        return jsonify({"status": "error",
                        "msg"   : "Could not generate report"}), 500
    return send_file(filepath,
                     as_attachment=True,
                     download_name=os.path.basename(filepath))


@api.route("/api/report/<int:session_id>/csv", methods=["GET"])
def api_export_csv(session_id):
    """Download CSV attendance report for a session."""
    filepath = generate_csv(session_id)
    if not filepath:
        return jsonify({"status": "error",
                        "msg"   : "Could not generate report"}), 500
    return send_file(filepath,
                     as_attachment=True,
                     download_name=os.path.basename(filepath))


# ══════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════

@api.route("/api/health", methods=["GET"])
def health():
    """Simple health check endpoint."""
    students = Student.query.count()
    sessions = Session.query.count()
    return jsonify({
        "status"          : "ok",
        "message"         : "Smart Attendance API running",
        "total_students"  : students,
        "total_sessions"  : sessions,
    })

# Add these routes at bottom of routes.py
@api.route("/login", methods=["GET"])
def login_page():
    if flask_session.get("faculty_id"):
        from flask import redirect
        return redirect("/")
    return render_template("login.html", error=None)


@api.route("/login", methods=["POST"])
def login_post():
    from flask import request, redirect
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    # Default admin account check
    if username == "admin" and password == "admin123":
        flask_session["faculty_id"]   = 0
        flask_session["faculty_name"] = "Admin"
        return redirect("/")

    # DB faculty check
    faculty = Faculty.query.filter_by(username=username).first()
    if faculty and bcrypt.check_password_hash(
            faculty.password, password):
        flask_session["faculty_id"]   = faculty.id
        flask_session["faculty_name"] = faculty.full_name
        return redirect("/")

    return render_template("login.html",
                            error="Invalid username or password")


@api.route("/logout")
def logout():
    from flask import redirect
    flask_session.clear()
    return redirect("/login")


@api.route("/api/current-faculty", methods=["GET"])
def current_faculty():
    return {
        "faculty_id"  : flask_session.get("faculty_id"),
        "faculty_name": flask_session.get("faculty_name", "Guest")
    }