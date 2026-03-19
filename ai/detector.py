# ai/detector.py
# Module 1: Live camera face detection using OpenCV Haar Cascade
# Works on Python 3.14 with no extra dependencies beyond opencv-python

import cv2

# ── Load the pre-trained Haar Cascade face detector ──────────────
# This XML file ships with OpenCV — no download needed
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_faces(frame):
    """
    Detect all faces in a single BGR frame.

    Args:
        frame: numpy array (H x W x 3) in BGR format from OpenCV

    Returns:
        List of tuples (x, y, w, h) — bounding box for each face
    """
    # Convert to grayscale — Haar Cascade works on grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improve contrast to help detection in variable lighting
    gray = cv2.equalizeHist(gray)

    # Detect faces
    # scaleFactor=1.1  → how much image size is reduced at each scale
    # minNeighbors=5   → higher = fewer false detections
    # minSize=(60,60)  → ignore tiny detections
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # detectMultiScale returns empty tuple if no faces found
    if len(faces) == 0:
        return []

    return [(x, y, w, h) for (x, y, w, h) in faces]


def draw_faces(frame, faces, label="Face"):
    """
    Draw bounding boxes and labels on the frame.

    Args:
        frame : original BGR frame
        faces : list of (x, y, w, h) from detect_faces()
        label : text to show above each box

    Returns:
        Annotated frame (numpy array)
    """
    for (x, y, w, h) in faces:
        # Draw green rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw label background
        cv2.rectangle(frame, (x, y - 28), (x + w, y), (0, 255, 0), -1)

        # Put label text
        cv2.putText(frame, label, (x + 4, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame


# ── LIVE TEST — run this file directly to test your camera ───────
if __name__ == "__main__":
    print("[INFO] Starting camera... Press Q to quit")
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check if webcam is connected.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        # Detect faces in current frame
        faces = detect_faces(frame)

        # Draw boxes on frame
        label = f"Face ({len(faces)} found)"
        frame = draw_faces(frame, faces, label)

        # Show FPS counter top-left
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        # Display the frame
        cv2.imshow("Smart Attendance - Face Detector", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera closed.")