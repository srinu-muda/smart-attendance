# app/models.py
# Database models + attendance engine
# Handles: sessions, marking present, duplicate prevention

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Student(db.Model):
    """Registered student master record."""
    __tablename__ = "students"

    id          = db.Column(db.Integer, primary_key=True)
    student_id  = db.Column(db.String(20), unique=True, nullable=False)
    full_name   = db.Column(db.String(100), nullable=False)
    branch      = db.Column(db.String(50), default="CSE")
    year        = db.Column(db.Integer,    default=1)
    created_at  = db.Column(db.DateTime,   default=datetime.utcnow)

    # Relationship: one student → many attendance logs
    logs = db.relationship("AttendanceLog", backref="student",
                            lazy=True)

    def to_dict(self):
        return {
            "id"        : self.id,
            "student_id": self.student_id,
            "full_name" : self.full_name,
            "branch"    : self.branch,
            "year"      : self.year
        }


from flask_login import UserMixin

class Faculty(db.Model, UserMixin):
    """Faculty login account."""
    __tablename__ = "faculty"

    id           = db.Column(db.Integer, primary_key=True)
    username     = db.Column(db.String(50), unique=True,
                              nullable=False)
    full_name    = db.Column(db.String(100), nullable=False)
    password     = db.Column(db.String(200), nullable=False)
    department   = db.Column(db.String(50), default="CSE")
    created_at   = db.Column(db.DateTime,
                              default=datetime.utcnow)

    def to_dict(self):
        return {
            "id"        : self.id,
            "username"  : self.username,
            "full_name" : self.full_name,
            "department": self.department
        }
    
class Session(db.Model):
    """A single class/lecture session."""
    __tablename__ = "sessions"

    id           = db.Column(db.Integer, primary_key=True)
    subject_code = db.Column(db.String(20),  nullable=False)
    subject_name = db.Column(db.String(100), default="")
    faculty_name = db.Column(db.String(100), default="")
    start_time   = db.Column(db.DateTime,    default=datetime.utcnow)
    end_time     = db.Column(db.DateTime,    nullable=True)
    is_active    = db.Column(db.Boolean,     default=True)

    # Relationship: one session → many attendance logs
    logs = db.relationship("AttendanceLog", backref="session",
                            lazy=True)

    def to_dict(self):
        return {
            "id"          : self.id,
            "subject_code": self.subject_code,
            "subject_name": self.subject_name,
            "faculty_name": self.faculty_name,
            "start_time"  : self.start_time.strftime("%Y-%m-%d %H:%M"),
            "end_time"    : self.end_time.strftime("%Y-%m-%d %H:%M")
                            if self.end_time else None,
            "is_active"   : self.is_active,
            "total_present": len(self.logs)
        }


class AttendanceLog(db.Model):
    """One attendance entry per student per session."""
    __tablename__ = "attendance_logs"

    id          = db.Column(db.Integer,  primary_key=True)
    student_id  = db.Column(db.String(20), db.ForeignKey("students.student_id"),
                             nullable=False)
    session_id  = db.Column(db.Integer,  db.ForeignKey("sessions.id"),
                             nullable=False)
    timestamp   = db.Column(db.DateTime, default=datetime.utcnow)
    confidence  = db.Column(db.Float,    default=0.0)
    status      = db.Column(db.String(10), default="PRESENT")

    # Prevent duplicate: same student cannot be marked twice per session
    __table_args__ = (
        db.UniqueConstraint("student_id", "session_id",
                            name="unique_attendance"),
    )

    def to_dict(self):
        return {
            "id"        : self.id,
            "student_id": self.student_id,
            "name"      : self.student.full_name
                          if self.student else "Unknown",
            "session_id": self.session_id,
            "timestamp" : self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "confidence": round(self.confidence, 3),
            "status"    : self.status
        }