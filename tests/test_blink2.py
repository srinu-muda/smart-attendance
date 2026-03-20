# tests/test_blink2.py
# Deep diagnostic — finds exactly what's failing

import cv2
import numpy as np
import mediapipe as mp
import sys

print("[INFO] Python:", sys.version)
print("[INFO] OpenCV:", cv2.__version__)
print("[INFO] MediaPipe:", mp.__version__)

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

def ear(lms, idx, w, h):
    pts = [(lms[i].x * w, lms[i].y * h) for i in idx]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C) if C else 0.0

# ── Test which MediaPipe API works ────────────────────────────────
print("\n[TEST] Checking MediaPipe API...")
use_legacy = False
mesh       = None

try:
    test = mp.solutions.face_mesh
    mesh = test.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3)
    use_legacy = True
    print("[OK] Legacy API (mp.solutions) works!")
except Exception as e:
    print(f"[FAIL] Legacy API failed: {e}")

if not use_legacy:
    print("[TEST] Trying new Tasks API...")
    try:
        import urllib.request, os
        model_path = "face_landmarker.task"
        if not os.path.exists(model_path):
            print("[INFO] Downloading model (~30MB)...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/"
                "face_landmarker.task",
                model_path)
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=model_path),
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_tracking_confidence=0.3)
        mesh       = vision.FaceLandmarker\
                         .create_from_options(opts)
        use_legacy = False
        print("[OK] New Tasks API works!")
    except Exception as e:
        print(f"[FAIL] New API also failed: {e}")
        print("[ERROR] MediaPipe completely broken.")
        exit()

print(f"\n[INFO] Using: "
      f"{'Legacy' if use_legacy else 'New'} API")
print("[INFO] Opening camera...")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Camera failed to open!")
    exit()

print("[INFO] Camera OK!")
print("[INFO] Look at camera — checking detection...\n")

frame_count    = 0
detected_count = 0
min_ear        = 1.0
max_ear        = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w = frame.shape[:2]

    landmarks  = None
    ear_val    = 0.0
    detected   = False

    # ── Get landmarks ─────────────────────────────────────────────
    try:
        if use_legacy:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mesh.process(rgb)
            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark
                detected  = True
        else:
            import mediapipe as mp2
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp2.Image(
                image_format=mp2.ImageFormat.SRGB, data=rgb)
            result = mesh.detect(mp_img)
            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                detected  = True
    except Exception as e:
        cv2.putText(frame, f"ERROR: {e}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

    # ── Compute EAR if detected ───────────────────────────────────
    if detected and landmarks:
        detected_count += 1
        try:
            l = ear(landmarks, LEFT_EYE,  w, h)
            r = ear(landmarks, RIGHT_EYE, w, h)
            ear_val = (l + r) / 2.0
            min_ear = min(min_ear, ear_val)
            max_ear = max(max_ear, ear_val)
        except Exception as e:
            cv2.putText(frame, f"EAR ERROR: {e}",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)

    # ── Draw diagnostics ──────────────────────────────────────────
    # Detection status
    det_color = (0, 200, 0) if detected else (0, 0, 255)
    det_text  = "FACE DETECTED" if detected \
                else "NO FACE DETECTED"
    cv2.putText(frame, det_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, det_color, 2)

    # EAR value
    if ear_val > 0:
        ear_color = (0, 0, 255) if ear_val < 0.20 \
                    else (0, 200, 0)
        cv2.putText(frame,
                    f"EAR = {ear_val:.4f}",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, ear_color, 3)
        cv2.putText(frame,
                    f"LEFT={l:.3f}  RIGHT={r:.3f}",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (200, 200, 200), 1)
    else:
        cv2.putText(frame,
                    "EAR = not computed",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 165, 255), 2)

    # Min/Max
    cv2.putText(frame,
                f"Min(blink)={min_ear:.3f}  "
                f"Max(open)={max_ear:.3f}",
                (20, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 0), 2)

    # Detection rate
    rate = (detected_count / frame_count * 100) \
           if frame_count > 0 else 0
    cv2.putText(frame,
                f"Detection rate: {rate:.0f}%  "
                f"Frames: {frame_count}",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (150, 150, 150), 1)

    # Draw landmarks if detected
    if detected and landmarks:
        # Draw eye points
        for idx in LEFT_EYE + RIGHT_EYE:
            lm = landmarks[idx]
            px = int(lm.x * w)
            py = int(lm.y * h)
            cv2.circle(frame, (px, py), 2,
                       (0, 255, 255), -1)

    # EAR bar
    if ear_val > 0:
        bar = int(min(ear_val / 0.5, 1.0) * 400)
        cv2.rectangle(frame,
                      (20, 220), (420, 240),
                      (50, 50, 50), -1)
        cv2.rectangle(frame,
                      (20, 220),
                      (20 + bar, 240),
                      ear_color, -1)

    cv2.putText(frame,
                "BLINK SLOWLY  |  Q = quit + results",
                (20, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1)

    cv2.imshow("Blink Diagnostic", frame)

    # Print to terminal every 20 frames
    if frame_count % 20 == 0:
        print(f"  Frame {frame_count:4d} | "
              f"Detected: {'YES' if detected else 'NO ':3s} | "
              f"EAR: {ear_val:.4f} | "
              f"Min: {min_ear:.4f} | "
              f"Max: {max_ear:.4f}")

    if cv2.waitKey(1) in [ord('q'), ord('Q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*50)
print("  DIAGNOSTIC RESULTS")
print("="*50)
print(f"  Total frames    : {frame_count}")
print(f"  Detected frames : {detected_count}")
print(f"  Detection rate  : "
      f"{detected_count/frame_count*100:.1f}%"
      if frame_count > 0 else "N/A")
print(f"  Eye OPEN  (max) : {max_ear:.4f}")
print(f"  Eye CLOSED (min): {min_ear:.4f}")
if max_ear > 0 and min_ear < 1.0:
    threshold = (min_ear + max_ear) / 2
    print(f"  Best threshold  : {threshold:.4f}")
print("="*50)
print("\nPaste ALL the terminal output here!")


# ## ▶️ Run It```
# python tests\test_blink2.py