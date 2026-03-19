# ai/liveness.py
# Module 2: Liveness Detection — compatible with MediaPipe 0.10+

import cv2
import numpy as np
import mediapipe as mp

# ── MediaPipe Face Mesh Setup (new API) ──────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ── Eye Landmark Indices (MediaPipe 478-point model) ─────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ── Thresholds ────────────────────────────────────────────────────
EAR_THRESHOLD = 0.22
CONSEC_FRAMES = 2
BLINKS_NEEDED = 2


def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
    """
    Compute Eye Aspect Ratio (EAR) for one eye.
    EAR close to 0 = eye closed, EAR ~0.25+ = eye open
    """
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        points.append((lm.x * image_w, lm.y * image_h))

    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


class LivenessChecker:
    """
    Detects real blinks to verify the person is LIVE (not a photo).
    Uses MediaPipe FaceMesh with the legacy mp.solutions fallback.
    """

    def __init__(self):
        self.blink_counter = 0
        self.total_blinks  = 0
        self.is_live       = False

        # ── Use legacy solutions API if available, else new API ──
        try:
            # Try old API first (mediapipe < 0.10)
            mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_legacy = True
            print("[INFO] Using MediaPipe legacy solutions API")
        except AttributeError:
            # New API (mediapipe >= 0.10) — use FaceLandmarker
            self.face_mesh = self._init_new_api()
            self.use_legacy = False
            print("[INFO] Using MediaPipe new tasks API")

    def _init_new_api(self):
        """Initialize FaceMesh using new MediaPipe Tasks API."""
        import urllib.request, os
        model_path = "face_landmarker.task"
        if not os.path.exists(model_path):
            print("[INFO] Downloading face landmarker model (~30 MB)...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("[INFO] Model downloaded!")
        base_opts   = mp_python.BaseOptions(model_asset_path=model_path)
        options     = vision.FaceLandmarkerOptions(
            base_options=base_opts,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        return vision.FaceLandmarker.create_from_options(options)

    def reset(self):
        """Reset state before checking a new student."""
        self.blink_counter = 0
        self.total_blinks  = 0
        self.is_live       = False

    def _get_landmarks_legacy(self, frame):
        """Get landmarks using old MediaPipe API."""
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        if result.multi_face_landmarks:
            return result.multi_face_landmarks[0].landmark
        return None

    def _get_landmarks_new(self, frame):
        """Get landmarks using new MediaPipe Tasks API."""
        import mediapipe as mp
        rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result     = self.face_mesh.detect(mp_image)
        if result.face_landmarks:
            return result.face_landmarks[0]
        return None

    def check(self, frame):
        """
        Process one frame, update blink count, return liveness verdict.

        Returns:
            (is_live, ear, total_blinks, annotated_frame)
        """
        h, w = frame.shape[:2]
        ear  = 0.0

        # Get landmarks using whichever API is available
        if self.use_legacy:
            landmarks = self._get_landmarks_legacy(frame)
        else:
            landmarks = self._get_landmarks_new(frame)

        if landmarks:
            left_ear  = eye_aspect_ratio(landmarks, LEFT_EYE,  w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear       = (left_ear + right_ear) / 2.0

            # ── Blink Detection Logic ──────────────────────────
            if ear < EAR_THRESHOLD:
                self.blink_counter += 1       # Eye closing
            else:
                if self.blink_counter >= CONSEC_FRAMES:
                    self.total_blinks += 1    # Blink complete!
                self.blink_counter = 0

        # ── Liveness Verdict ──────────────────────────────────────
        if self.total_blinks >= BLINKS_NEEDED:
            self.is_live = True

        # ── Annotate Frame ────────────────────────────────────────
        blinks_left  = max(0, BLINKS_NEEDED - self.total_blinks)
        status_color = (0, 255, 0) if ear >= EAR_THRESHOLD else (0, 0, 255)

        if self.is_live:
            msg   = "✓ LIVE - REAL PERSON VERIFIED"
            color = (0, 255, 0)
            cv2.rectangle(frame, (0, 0),
                          (frame.shape[1], frame.shape[0]), (0, 255, 0), 8)
        else:
            msg   = f"Please blink {blinks_left} more time(s)"
            color = (0, 165, 255)

        cv2.putText(frame, msg,           (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Blinks: {self.total_blinks}/{BLINKS_NEEDED}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return self.is_live, ear, self.total_blinks, frame


# ── LIVE TEST ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] Liveness Detection Test — blink twice to verify")
    print("[INFO] Press Q to quit")

    cap     = cv2.VideoCapture(0)
    checker = LivenessChecker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        is_live, ear, blinks, frame = checker.check(frame)
        cv2.imshow("Liveness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n[RESULT]", "✅ LIVE PERSON" if checker.is_live else "❌ NOT VERIFIED")
    
#     What you should see:**
# - Orange text: `"Blink 2 more time(s) to verify"`
# - Blink naturally twice → text turns **green**: `"LIVE - REAL PERSON"`
# - Green border appears around the whole frame ✅
# - Try holding a **photo** to the camera → it should NOT reach 2 blinks
