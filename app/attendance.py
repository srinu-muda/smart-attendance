# app/attendance.py
# Attendance business logic
# mark_present()     — mark a student, prevent duplicates
# get_session_report() — fetch all logs for a session
# sync_students()    — sync enrolled faces → Student DB table

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from app.models import db, Student, Session, AttendanceLog
from ai.encoder import load_db as load_face_db


def sync_students():
    """
    Sync face-enrolled students from encodings.pkl
    into the Student database table.
    Call this once at app startup.
    """
    face_db = load_face_db()
    synced  = 0

    for student_id, info in face_db.items():
        # Check if student already exists in DB
        existing = Student.query.filter_by(
                       student_id=student_id).first()
        if not existing:
            student = Student(
                student_id = student_id,
                full_name  = info["name"]
            )
            db.session.add(student)
            synced += 1

    db.session.commit()
    if synced:
        print(f"[DB] Synced {synced} new student(s) to database.")


def start_session(subject_code, subject_name="",
                  faculty_name=""):
    """
    Start a new attendance session.
    Closes any previously active session first.

    Returns:
        Session object
    """
    # Close any open sessions
    open_sessions = Session.query.filter_by(is_active=True).all()
    for s in open_sessions:
        s.is_active = False
        s.end_time  = datetime.utcnow()
    db.session.commit()

    # Create new session
    session = Session(
        subject_code = subject_code,
        subject_name = subject_name,
        faculty_name = faculty_name,
        start_time   = datetime.utcnow(),
        is_active    = True
    )
    db.session.add(session)
    db.session.commit()

    print(f"[SESSION] Started: {subject_code} "
          f"(ID: {session.id})")
    return session


def stop_session(session_id):
    """
    Stop an active session and return summary.

    Returns:
        dict with session stats
    """
    session = Session.query.get(session_id)
    if not session:
        return {"error": "Session not found"}

    session.is_active = False
    session.end_time  = datetime.utcnow()
    db.session.commit()

    total_students = Student.query.count()
    total_present  = AttendanceLog.query.filter_by(
                         session_id=session_id).count()

    print(f"[SESSION] Stopped: {session.subject_code} | "
          f"Present: {total_present}/{total_students}")

    return {
        "session_id"    : session_id,
        "subject"       : session.subject_code,
        "total_students": total_students,
        "total_present" : total_present,
        "total_absent"  : total_students - total_present,
        "start_time"    : session.start_time.strftime(
                              "%Y-%m-%d %H:%M"),
        "end_time"      : session.end_time.strftime(
                              "%Y-%m-%d %H:%M")
    }


def get_active_session():
    """Return the currently active session or None."""
    return Session.query.filter_by(is_active=True).first()


def mark_present(student_id, session_id, confidence=1.0):
    """
    Mark a student as present in a session.
    Silently ignores duplicates (same student, same session).

    Args:
        student_id : e.g. "U24CS167"
        session_id : integer session ID
        confidence : ArcFace similarity score

    Returns:
        dict with status and message
    """
    # Check student exists
    student = Student.query.filter_by(
                  student_id=student_id).first()
    if not student:
        return {"status": "error",
                "msg"   : f"Student {student_id} not in DB"}

    # Check session exists and is active
    session = Session.query.get(session_id)
    if not session:
        return {"status": "error",
                "msg"   : "Session not found"}
    if not session.is_active:
        return {"status": "error",
                "msg"   : "Session is not active"}

    # Check for duplicate
    existing = AttendanceLog.query.filter_by(
                   student_id=student_id,
                   session_id=session_id).first()
    if existing:
        return {"status": "duplicate",
                "msg"   : f"{student.full_name} already marked"}

    # Mark present
    log = AttendanceLog(
        student_id = student_id,
        session_id = session_id,
        confidence = confidence,
        status     = "PRESENT",
        timestamp  = datetime.utcnow()
    )
    db.session.add(log)
    db.session.commit()

    print(f"  [DB] Marked PRESENT: {student.full_name} "
          f"({student_id})  conf={confidence:.3f}")

    return {"status" : "ok",
            "msg"    : f"{student.full_name} marked present",
            "log_id" : log.id}


def get_session_report(session_id):
    """
    Get full attendance report for a session.

    Returns:
        dict with present list, absent list, stats
    """
    session = Session.query.get(session_id)
    if not session:
        return {"error": "Session not found"}

    all_students    = Student.query.all()
    present_logs    = AttendanceLog.query.filter_by(
                          session_id=session_id).all()
    present_ids     = {log.student_id for log in present_logs}

    present_list = []
    absent_list  = []

    for student in all_students:
        if student.student_id in present_ids:
            log = next(l for l in present_logs
                       if l.student_id == student.student_id)
            present_list.append({
                "student_id": student.student_id,
                "name"      : student.full_name,
                "time"      : log.timestamp.strftime("%H:%M:%S"),
                "confidence": round(log.confidence, 3),
                "status"    : "PRESENT"
            })
        else:
            absent_list.append({
                "student_id": student.student_id,
                "name"      : student.full_name,
                "time"      : "—",
                "confidence": 0,
                "status"    : "ABSENT"
            })

    return {
        "session"      : session.to_dict(),
        "present"      : present_list,
        "absent"       : absent_list,
        "total_present": len(present_list),
        "total_absent" : len(absent_list),
        "total"        : len(all_students)
    }