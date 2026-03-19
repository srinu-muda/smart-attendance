# ai/encoder.py
# Module 3: Face Enrollment
# Captures face photos for a student and saves their face embedding (fingerprint)
# Uses OpenCV + ONNX (no dlib/tensorflow needed — works on Python 3.14)

import cv2
import os
import pickle
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────
STUDENTS_DIR   = "data/students"      # Folder to save face images
ENCODINGS_FILE = "data/encodings.pkl" # Saved face embeddings
PHOTOS_NEEDED  = 10                   # Photos captured per student

# ── Load Haar Cascade for face detection during capture ───────────
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def get_face_embedding(face_img):
    """
    Generate a simple but effective face embedding using
    histogram of pixel values + resize normalization.

    This is our lightweight embedding that works without
    dlib/tensorflow on Python 3.14.

    Args:
        face_img : cropped face image (any size, BGR)

    Returns:
        numpy array of shape (1024,) — the face fingerprint
    """
    # Step 1: Resize to fixed 64x64
    face = cv2.resize(face_img, (64, 64))

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply histogram equalization (normalize lighting)
    gray = cv2.equalizeHist(gray)

    # Step 4: Flatten pixel values and normalize to 0-1
    pixels = gray.flatten().astype(np.float32) / 255.0

    # Step 5: Compute Local Binary Pattern (LBP) histogram
    # LBP captures texture patterns — works well for faces
    lbp = compute_lbp(gray)

    # Step 6: Combine pixel features + LBP histogram
    embedding = np.concatenate([pixels, lbp])

    return embedding


def compute_lbp(gray_img):
    """
    Compute a simplified Local Binary Pattern histogram.
    LBP is great for face texture — robust to lighting changes.

    Args:
        gray_img : 64x64 grayscale image

    Returns:
        numpy array of shape (256,) — LBP histogram
    """
    h, w   = gray_img.shape
    lbp    = np.zeros((h, w), dtype=np.uint8)

    # Compare each pixel with its 8 neighbors
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = gray_img[i, j]
            code   = 0
            code |= (gray_img[i-1, j-1] >= center) << 7
            code |= (gray_img[i-1, j  ] >= center) << 6
            code |= (gray_img[i-1, j+1] >= center) << 5
            code |= (gray_img[i,   j+1] >= center) << 4
            code |= (gray_img[i+1, j+1] >= center) << 3
            code |= (gray_img[i+1, j  ] >= center) << 2
            code |= (gray_img[i+1, j-1] >= center) << 1
            code |= (gray_img[i,   j-1] >= center) << 0
            lbp[i, j] = code

    # Return normalized histogram (256 bins)
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist    = hist.astype(np.float32)
    hist   /= (hist.sum() + 1e-6)   # Normalize
    return hist


def enroll_student(student_id, student_name):
    """
    Capture face photos from webcam and save embedding for a student.

    Args:
        student_id   : unique ID e.g. "U24CS167"
        student_name : full name e.g. "Maneesh Kumar"
    """
    # Create folder for this student
    student_dir = os.path.join(STUDENTS_DIR, student_id)
    os.makedirs(student_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print(f"\n[ENROLL] Enrolling: {student_name} ({student_id})")
    print(f"[ENROLL] We will capture {PHOTOS_NEEDED} photos.")
    print("[ENROLL] Look at the camera. Press SPACE to capture each photo.")
    print("[ENROLL] Press Q to cancel.\n")

    cap          = cv2.VideoCapture(0)
    captured     = 0
    embeddings   = []

    while captured < PHOTOS_NEEDED:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces   = FACE_CASCADE.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        face_found = False

        for (x, y, w, h) in faces:
            face_found = True
            # Draw box around detected face
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Show capture count inside box
            cv2.putText(display, f"Photo {captured+1}/{PHOTOS_NEEDED}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        # Instructions overlay
        status = "Face detected - Press SPACE to capture" if face_found \
                 else "No face found - adjust position"
        color  = (0, 255, 0) if face_found else (0, 0, 255)

        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, f"Student: {student_name} ({student_id})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display, f"Captured: {captured}/{PHOTOS_NEEDED}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Face Enrollment", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[ENROLL] Cancelled.")
            break

        if key == ord(' ') and face_found and len(faces) > 0:
            # Capture the first detected face
            x, y, w, h   = faces[0]
            face_crop    = frame[y:y+h, x:x+w]

            # Save photo to disk
            photo_path = os.path.join(student_dir, f"{captured+1}.jpg")
            cv2.imwrite(photo_path, face_crop)

            # Generate embedding for this photo
            emb = get_face_embedding(face_crop)
            embeddings.append(emb)

            captured += 1
            print(f"  [✓] Photo {captured}/{PHOTOS_NEEDED} saved → {photo_path}")

            # Flash green screen as feedback
            green = np.zeros_like(display)
            green[:] = (0, 255, 0)
            cv2.imshow("Face Enrollment", green)
            cv2.waitKey(200)

    cap.release()
    cv2.destroyAllWindows()

    # ── Save mean embedding to encodings file ─────────────────────
    if len(embeddings) >= 3:
        mean_embedding = np.mean(embeddings, axis=0)

        # Load existing encodings or start fresh
        if os.path.exists(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, 'rb') as f:
                db = pickle.load(f)
        else:
            db = {}

        db[student_id] = {
            "name"     : student_name,
            "embedding": mean_embedding
        }

        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump(db, f)

        print(f"\n[✓] Enrollment complete for {student_name}!")
        print(f"[✓] {len(embeddings)} photos used to build face fingerprint.")
        print(f"[✓] Saved to {ENCODINGS_FILE}")
    else:
        print("[✗] Not enough photos captured. Please try again.")


def list_enrolled():
    """Show all currently enrolled students."""
    if not os.path.exists(ENCODINGS_FILE):
        print("[INFO] No students enrolled yet.")
        return

    with open(ENCODINGS_FILE, 'rb') as f:
        db = pickle.load(f)

    print(f"\n[INFO] Enrolled Students ({len(db)} total):")
    print("-" * 40)
    for sid, info in db.items():
        print(f"  ID: {sid}  |  Name: {info['name']}")
    print("-" * 40)


# ── ENROLLMENT MENU ───────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   SMART ATTENDANCE — Face Enrollment Module")
    print("=" * 50)

    while True:
        print("\n[1] Enroll new student")
        print("[2] List enrolled students")
        print("[3] Exit")
        choice = input("\nEnter choice (1/2/3): ").strip()

        if choice == "1":
            sid  = input("Enter Student ID (e.g. U24CS167): ").strip()
            name = input("Enter Student Name: ").strip()
            enroll_student(sid, name)

        elif choice == "2":
            list_enrolled()

        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")
