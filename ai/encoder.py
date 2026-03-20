# ai/encoder.py — UPGRADED with InsightFace ArcFace + FAISS
# Uses buffalo_sc model — optimized for CPU, 97% accuracy
# Generates 512-d ArcFace embeddings stored in FAISS index

import cv2
import os
import sys
import pickle
import numpy as np
import faiss

# ── Fix module path so it works both ways ─────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── InsightFace setup ─────────────────────────────────────────────
import insightface
from insightface.app import FaceAnalysis

# ── Paths ─────────────────────────────────────────────────────────
STUDENTS_DIR   = "data/students"
ENCODINGS_FILE = "data/encodings.pkl"   # student_id → name mapping
FAISS_INDEX    = "data/faiss.index"     # FAISS vector index
LABELS_FILE    = "data/labels.pkl"      # index position → student_id
PHOTOS_NEEDED  = 15                     # photos per student

# ── Global face app (loaded once, reused) ─────────────────────────
_app = None

def get_face_app():
    """
    Load InsightFace buffalo_sc model once and reuse.
    buffalo_sc = small + fast, optimized for CPU.
    First call downloads ~100 MB model automatically.
    """
    global _app
    if _app is None:
        print("[INFO] Loading InsightFace buffalo_sc model...")
        print("[INFO] First run may download model — please wait...")
        _app = FaceAnalysis(
            name="buffalo_sc",          # Small + fast for CPU
            providers=["CPUExecutionProvider"]
        )
        _app.prepare(ctx_id=0, det_size=(320, 320))  # Smaller = faster
        print("[INFO] Model loaded successfully!")
    return _app


def get_embedding(img_bgr):
    """
    Get 512-d ArcFace embedding for a face image.

    Args:
        img_bgr : BGR image (full frame or face crop)

    Returns:
        numpy array (512,) normalized embedding
        or None if no face detected
    """
    app   = get_face_app()
    faces = app.get(img_bgr)

    if not faces:
        return None

    # Use the largest detected face
    face      = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) *
                                         (f.bbox[3]-f.bbox[1]))
    embedding = face.embedding

    if embedding is None:
        return None

    # L2 normalize so cosine similarity = dot product
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.astype(np.float32)


def build_faiss_index():
    """
    Build FAISS index from all stored student photos.
    Called automatically after every enrollment/update/delete.

    FAISS IndexFlatIP = exact inner product search
    = cosine similarity on L2-normalized vectors
    Same person    → score ~0.85–0.99
    Diff person    → score ~0.10–0.40
    """
    db = load_db()
    if not db:
        print("[WARN] No students enrolled — skipping index build.")
        return

    print("\n[FAISS] Building search index...")

    all_embeddings = []   # Will become shape (N, 512)
    all_labels     = []   # student_id for each embedding row

    app = get_face_app()

    for student_id, info in db.items():
        student_dir = os.path.join(STUDENTS_DIR, student_id)
        if not os.path.exists(student_dir):
            continue

        photo_files = [f for f in os.listdir(student_dir)
                       if f.lower().endswith('.jpg')]
        count = 0

        for photo_file in photo_files:
            path = os.path.join(student_dir, photo_file)
            img  = cv2.imread(path)
            if img is None:
                continue

            # Get embedding directly from full frame
            faces = app.get(img)
            if not faces:
                continue

            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) *
                                             (f.bbox[3]-f.bbox[1]))
            emb = face.embedding
            if emb is None:
                continue

            emb = emb / np.linalg.norm(emb)
            all_embeddings.append(emb.astype(np.float32))
            all_labels.append(student_id)
            count += 1

        print(f"  [✓] {info['name']:20s} → {count} embeddings indexed")

    if not all_embeddings:
        print("[ERROR] No valid embeddings found. "
              "Make sure photos contain clear faces.")
        return

    # Stack into matrix and build FAISS index
    matrix = np.stack(all_embeddings).astype(np.float32)
    index  = faiss.IndexFlatIP(512)   # Inner Product on L2-norm = cosine
    index.add(matrix)

    # Save index and labels to disk
    faiss.write_index(index, FAISS_INDEX)
    with open(LABELS_FILE, 'wb') as f:
        pickle.dump(all_labels, f)

    print(f"\n[✓] FAISS index built successfully!")
    print(f"    Vectors : {index.ntotal}")
    print(f"    Students: {len(db)}")
    print(f"    Saved to: {FAISS_INDEX}")


def load_db():
    """Load student ID → name mapping from disk."""
    if not os.path.exists(ENCODINGS_FILE):
        return {}
    with open(ENCODINGS_FILE, 'rb') as f:
        return pickle.load(f)


def save_db(db):
    """Save student ID → name mapping to disk."""
    os.makedirs("data", exist_ok=True)
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(db, f)


def enroll_student(student_id, student_name):
    """
    Capture photos from webcam and enroll a new student.
    Saves full frames — InsightFace works better on full frames.
    Rebuilds FAISS index after enrollment.
    """
    student_dir = os.path.join(STUDENTS_DIR, student_id)
    os.makedirs(student_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print(f"\n[ENROLL] Enrolling: {student_name} ({student_id})")
    print(f"[INFO]   Capturing {PHOTOS_NEEDED} photos.")
    print("[INFO]   TIPS for best ArcFace accuracy:")
    print("         Photos 1-5  : look straight at camera")
    print("         Photos 6-8  : turn slightly left")
    print("         Photos 9-11 : turn slightly right")
    print("         Photos 12-13: look up slightly")
    print("         Photos 14-15: different expressions")
    print("[INFO]   Press SPACE to capture | Q to cancel\n")

    # Load model before opening camera
    app = get_face_app()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    captured = 0

    while captured < PHOTOS_NEEDED:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from camera.")
            break

        display = frame.copy()

        # Detect faces using InsightFace RetinaFace
        faces      = app.get(frame)
        face_found = len(faces) > 0

        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            # Clamp to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display,
                        f"Photo {captured+1}/{PHOTOS_NEEDED}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Status overlay
        if face_found:
            msg   = "Face detected — Press SPACE to capture"
            color = (0, 255, 0)
        else:
            msg   = "No face detected — adjust position"
            color = (0, 0, 255)

        cv2.putText(display, msg,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(display,
                    f"{student_name} ({student_id})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2)

        # Progress bar
        progress = int((captured / PHOTOS_NEEDED) * 400)
        cv2.rectangle(display, (10, 80), (410, 100), (50, 50, 50), -1)
        cv2.rectangle(display, (10, 80),
                      (10 + progress, 100), (0, 200, 0), -1)
        cv2.putText(display, f"{captured}/{PHOTOS_NEEDED}",
                    (415, 95), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)

        cv2.imshow("Face Enrollment - ArcFace", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[ENROLL] Cancelled by user.")
            break

        if key == ord(' ') and face_found:
            # Save full frame (not just crop)
            photo_path = os.path.join(student_dir, f"{captured+1}.jpg")
            cv2.imwrite(photo_path, frame)
            captured += 1
            print(f"  [✓] Photo {captured}/{PHOTOS_NEEDED} saved")

            # Green flash feedback
            flash = np.zeros_like(display)
            flash[:] = (0, 200, 0)
            cv2.imshow("Face Enrollment - ArcFace", flash)
            cv2.waitKey(150)

    cap.release()
    cv2.destroyAllWindows()

    if captured >= 5:
        # Save student to DB
        db = load_db()
        db[student_id] = {"name": student_name}
        save_db(db)
        print(f"\n[✓] Enrollment complete for {student_name}!")
        print(f"    Photos saved: {captured}")

        # Rebuild FAISS index with new student
        build_faiss_index()
    else:
        print("[✗] Not enough photos captured. "
              "Need at least 5. Please try again.")


def delete_student(student_id):
    """Remove a student from DB + photos and rebuild index."""
    db = load_db()
    if student_id not in db:
        print(f"[ERROR] Student ID '{student_id}' not found.")
        return

    name = db.pop(student_id)['name']
    save_db(db)

    # Delete their photos
    import shutil
    student_dir = os.path.join(STUDENTS_DIR, student_id)
    if os.path.exists(student_dir):
        shutil.rmtree(student_dir)

    print(f"[✓] Deleted {name} ({student_id}) and all their photos.")

    # Rebuild index without this student
    build_faiss_index()


def update_student(student_id, student_name):
    """Delete old enrollment and re-enroll with fresh photos."""
    import shutil

    # Remove from DB
    db = load_db()
    db.pop(student_id, None)
    save_db(db)

    # Delete old photos
    student_dir = os.path.join(STUDENTS_DIR, student_id)
    if os.path.exists(student_dir):
        shutil.rmtree(student_dir)

    print(f"[INFO] Old data cleared. Starting fresh enrollment...")

    # Fresh enrollment
    enroll_student(student_id, student_name)


def list_enrolled():
    """Display all enrolled students with photo counts."""
    db = load_db()
    if not db:
        print("[INFO] No students enrolled yet.")
        return

    faiss_exists = os.path.exists(FAISS_INDEX)
    print(f"\n{'─'*55}")
    print(f"  Enrolled Students ({len(db)} total)")
    print(f"  FAISS Index: {'✓ Built' if faiss_exists else '✗ Not built'}")
    print(f"{'─'*55}")
    for sid, info in db.items():
        student_dir = os.path.join(STUDENTS_DIR, sid)
        photos = len([f for f in os.listdir(student_dir)
                      if f.endswith('.jpg')]) \
                 if os.path.exists(student_dir) else 0
        print(f"  {sid:15s}  {info['name']:25s}  {photos} photos")
    print(f"{'─'*55}")


# ── MAIN MENU ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   SMART ATTENDANCE — ArcFace Enrollment Module")
    print("   Model: InsightFace buffalo_sc (CPU optimized)")
    print("=" * 55)

    while True:
        print("\n[1] Enroll new student")
        print("[2] List enrolled students")
        print("[3] Update / re-enroll student")
        print("[4] Delete student")
        print("[5] Rebuild FAISS index")
        print("[6] Exit")

        choice = input("\nEnter choice: ").strip()

        if choice == "1":
            sid  = input("Student ID   : ").strip()
            name = input("Student Name : ").strip()
            enroll_student(sid, name)

        elif choice == "2":
            list_enrolled()

        elif choice == "3":
            list_enrolled()
            sid  = input("Student ID to update : ").strip()
            name = input("Student Name         : ").strip()
            update_student(sid, name)

        elif choice == "4":
            list_enrolled()
            sid = input("Student ID to delete : ").strip()
            delete_student(sid)

        elif choice == "5":
            build_faiss_index()

        elif choice == "6":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1-6.")