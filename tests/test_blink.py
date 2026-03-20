# tests/test_blink.py
# ISOLATED blink test — shows your exact EAR values in real time
# Run this FIRST to find your personal EAR threshold

import cv2
import numpy as np
import mediapipe as mp

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

def ear(lms, idx, w, h):
    pts = [(lms[i].x * w, lms[i].y * h) for i in idx]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C) if C else 0.0

print("[INFO] Starting blink test...")
print("[INFO] BLINK SLOWLY AND FULLY — watch terminal")
print("[INFO] Press Q to quit\n")

try:
    mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    use_legacy = True
    print("[INFO] MediaPipe legacy OK")
except:
    use_legacy = False
    print("[INFO] Using new MediaPipe API")

cap         = cv2.VideoCapture(0)
min_ear     = 1.0   # Track lowest EAR seen (during blink)
max_ear     = 0.0   # Track highest EAR seen (eye open)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w = frame.shape[:2]

    ear_val = 0.0

    if use_legacy:
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mesh.process(rgb)
        if result.multi_face_landmarks:
            lms     = result.multi_face_landmarks[0].landmark
            l_ear   = ear(lms, LEFT_EYE,  w, h)
            r_ear   = ear(lms, RIGHT_EYE, w, h)
            ear_val = (l_ear + r_ear) / 2.0

            min_ear = min(min_ear, ear_val)
            max_ear = max(max_ear, ear_val)

    # Color: red when eye closing, green when open
    if ear_val < 0.18:
        color = (0, 0, 255)      # Red = closed
        label = f"EYE CLOSED  EAR={ear_val:.3f}"
    elif ear_val < 0.22:
        color = (0, 165, 255)    # Orange = closing
        label = f"CLOSING     EAR={ear_val:.3f}"
    else:
        color = (0, 200, 0)      # Green = open
        label = f"EYE OPEN    EAR={ear_val:.3f}"

    # Big EAR display
    cv2.putText(frame, label,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, color, 3)

    # Min/Max tracker
    cv2.putText(frame,
                f"Min EAR (blink): {min_ear:.3f}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
    cv2.putText(frame,
                f"Max EAR (open) : {max_ear:.3f}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 200, 0), 2)

    # Instructions
    cv2.putText(frame,
                "BLINK SLOWLY AND FULLY several times",
                (20, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 0), 2)
    cv2.putText(frame,
                "Q to quit and see results",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1)

    # Live EAR bar
    bar_val = int(min(ear_val / 0.4, 1.0) * 400)
    cv2.rectangle(frame, (20, 170), (420, 195),
                  (50, 50, 50), -1)
    cv2.rectangle(frame, (20, 170),
                  (20 + bar_val, 195), color, -1)
    cv2.putText(frame, "EAR level",
                (20, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (150, 150, 150), 1)

    # Print to terminal every 15 frames
    if frame_count % 15 == 0 and ear_val > 0:
        print(f"  EAR={ear_val:.3f}  "
              f"min={min_ear:.3f}  "
              f"max={max_ear:.3f}")

    cv2.imshow("Blink Test - Find Your EAR", frame)

    if cv2.waitKey(1) in [ord('q'), ord('Q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*45)
print("  YOUR BLINK TEST RESULTS")
print("="*45)
print(f"  Eye OPEN  (max EAR) : {max_ear:.3f}")
print(f"  Eye CLOSED (min EAR): {min_ear:.3f}")
print(f"  Recommended threshold: "
      f"{(min_ear + max_ear) / 2:.3f}")
print("="*45)
print("\nPaste these numbers and I will fix the threshold!")

# ## ▶️ Run It
# ```
# python tests\test_blink.py
# ```

# - Blink **slowly and fully** 5 times
# - Watch the EAR value change on screen
# - Press Q when done

# **Paste the final output** — the 3 lines showing:
# ```
# Eye OPEN  (max EAR) : 0.XXX
# Eye CLOSED (min EAR): 0.XXX
# Recommended threshold: 0.XXX