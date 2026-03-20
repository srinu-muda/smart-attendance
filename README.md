# 🎓 Smart AI Attendance System

> Final Year Project — SVNIT Surat | Department of Computer Science & Engineering
> Developed by **Dasari Maneesh Kumar (U24CS167)** & **Srinu Muda (U24CS124)**

---

## 📌 Problem Statement

Traditional classroom attendance is manual, time-consuming, and vulnerable to proxy attendance (buddy punching). This system solves all three problems using AI-powered face recognition with liveness detection.

---

## 🚀 Features

- ✅ **Automatic Attendance** — detects and marks students present via webcam
- ✅ **Liveness Detection** — blink verification prevents photo spoofing
- ✅ **ArcFace Recognition** — 97%+ accuracy using InsightFace buffalo_sc model
- ✅ **FAISS Vector Search** — instant face matching, scales to 100+ students
- ✅ **Web Dashboard** — real-time attendance table, session control, alerts
- ✅ **Faculty Login** — secure login system per faculty
- ✅ **Export Reports** — one-click Excel and CSV export
- ✅ **SQLite Database** — zero-config local database
- ✅ **One Command Launch** — entire system starts with `python run.py`

---

## 🧠 AI Pipeline
```
Camera Feed
    │
    ▼
Face Detection (InsightFace RetinaFace)
    │
    ▼
Liveness Check (MediaPipe EAR Blink Detection)
    │ ← BLOCKED if no blink (photo spoof rejected)
    ▼
Face Recognition (ArcFace 512-d embedding)
    │
    ▼
FAISS Vector Search (cosine similarity)
    │
    ▼
Attendance Marked via Flask REST API
    │
    ▼
Dashboard Updated in Real-Time
```

---

## 🗂️ Project Structure
```
smart_attendance/
├── ai/
│   ├── detector.py        # Haar Cascade face detection
│   ├── encoder.py         # ArcFace enrollment + FAISS index
│   ├── liveness.py        # MediaPipe EAR blink detection
│   ├── recogniser.py      # ArcFace + FAISS recognition
│   └── pipeline.py        # Complete live pipeline
├── app/
│   ├── models.py          # SQLAlchemy DB models
│   ├── attendance.py      # Attendance business logic
│   ├── routes.py          # Flask REST API endpoints
│   └── report.py          # Excel/CSV report generator
├── templates/
│   ├── dashboard.html     # Main web dashboard
│   └── login.html         # Faculty login page
├── data/                  # Runtime data (git-ignored)
│   ├── attendance.db      # SQLite database
│   ├── faiss.index        # FAISS vector index
│   ├── encodings.pkl      # Student name mapping
│   ├── labels.pkl         # FAISS label mapping
│   └── students/          # Enrolled face photos
├── tests/
│   ├── test_attendance.py
│   ├── test_report.py
│   └── test_blink.py
├── run.py                 # Main entry point
├── config.yaml            # Configuration
└── requirements.txt
```

---

## ⚙️ Installation

### Prerequisites
- Windows 10/11
- Python 3.9–3.14
- Webcam
- Microsoft C++ Build Tools (for InsightFace)

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/smart-attendance.git
cd smart-attendance
```

### Step 2 — Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the system
```bash
python run.py
```

### Step 5 — Open dashboard
```
http://localhost:5000
```
Login: `admin` / `admin123`

---

## 📸 Enrolling Students
```bash
python -m ai.encoder
```

- Choose option 1 → Enroll new student
- Enter Student ID and Name
- Capture 15 photos with varied angles
- FAISS index rebuilds automatically

---

## 🎮 Usage Workflow
```
1. python run.py
2. Open http://localhost:5000
3. Login with admin / admin123
4. Click "Start Camera" button
5. Enter subject details → Start Session
6. Students face camera → blink twice → marked present
7. Stop Session → Export Excel/CSV
```

---

## 🏗️ System Architecture

| Layer | Technology |
|-------|-----------|
| Camera / Sensor | OpenCV VideoCapture |
| Face Detection | InsightFace RetinaFace |
| Liveness | MediaPipe FaceMesh (EAR) |
| Face Encoding | ArcFace 512-d (buffalo_sc) |
| Face Matching | FAISS IndexFlatIP |
| Backend API | Flask + SQLAlchemy |
| Database | SQLite |
| Frontend | Bootstrap 5 + Chart.js |
| Reports | pandas + openpyxl |

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Recognition Accuracy | ~97% (buffalo_sc model) |
| Known person score | ~0.85–0.92 |
| Unknown person score | ~0.10–0.35 |
| Liveness detection | Blink-based EAR |
| False positive rate | Near zero with threshold 0.65 |
| Processing speed | Every 4th frame (~7 FPS effective) |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | System health check |
| GET | `/api/students` | List all students |
| POST | `/api/session/start` | Start new session |
| POST | `/api/session/stop` | Stop active session |
| GET | `/api/session/active` | Get active session |
| POST | `/api/mark` | Mark student present |
| GET | `/api/attendance/live` | Live attendance data |
| GET | `/api/report/{id}/excel` | Download Excel report |
| GET | `/api/report/{id}/csv` | Download CSV report |
| GET | `/api/pipeline/status` | Pipeline running status |
| POST | `/api/pipeline/start` | Start camera pipeline |
| POST | `/api/pipeline/stop` | Stop camera pipeline |

---

## 🔮 Future Enhancements

- [ ] Cloud deployment (AWS / Google Cloud)
- [ ] Mobile app (React Native)
- [ ] Multi-classroom support
- [ ] Emotion / engagement analytics
- [ ] LMS integration (Moodle / Google Classroom)
- [ ] GPU acceleration for faster processing
- [ ] Depth sensor for stronger liveness

---

## 🛡️ Security Features

- **Liveness detection** — blink verification blocks photo attacks
- **Confirmation threshold** — requires 5 consecutive detections
- **Duplicate prevention** — one mark per student per session
- **Faculty login** — session-based authentication
- **Local storage** — all data stays on-premise

---

## 📄 License

This project is developed for academic purposes at SVNIT Surat.

---

## 👥 Authors

| Name | Roll Number |
|------|------------|
| Dasari Maneesh Kumar | U24CS167 |
| Srinu Muda | U24CS124 |

**Guide:** CISMR Group — Computational Intelligence & Smart Motion Robotics
**Institution:** SVNIT Surat — Department of Computer Science & Engineering
**Year:** 2026
```

---

## 📤 Final GitHub Push
```
git add .
git commit -m "Project complete: requirements.txt + README.md added"
git push
```

---

## ✅ Project Complete Checklist

Run through this before viva:
```
python run.py                    ← starts everything
http://localhost:5000            ← dashboard opens
admin / admin123                 ← login works
Start Camera button              ← pipeline starts
Start Session (CS501, AI, Prof)  ← session created
Face camera + blink 2x           ← marked present
Dashboard updates live           ← green PRESENT badge
Stop Session                     ← session ends
Export Excel                     ← file downloads
Past Sessions table              ← all sessions listed
