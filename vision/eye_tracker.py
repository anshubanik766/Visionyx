import cv2
import mediapipe as mp
import numpy as np


class EyeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Blink tracking
        self.blink_count = 0
        self.eye_closed = False
        self.prev_eye_closed = False

        # Tune this after debugging if needed
        self.EYE_THRESHOLD = 0.18

        # Eye landmark indices (MediaPipe Face Mesh)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def _distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _eye_aspect_ratio(self, landmarks, eye_points, w, h):
        # Convert landmarks to pixel coordinates
        pts = []
        for idx in eye_points:
            lm = landmarks[idx]
            pts.append((int(lm.x * w), int(lm.y * h)))

        # EAR formula (vertical / horizontal ratio)
        vertical1 = self._distance(pts[1], pts[5])
        vertical2 = self._distance(pts[2], pts[4])
        horizontal = self._distance(pts[0], pts[3])

        ear = (vertical1 + vertical2) / (2.0 * horizontal + 1e-6)
        return ear

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        h, w, _ = frame.shape

        blink_detected = False
        left_ear = right_ear = 0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_ear = self._eye_aspect_ratio(landmarks, self.LEFT_EYE, w, h)
            right_ear = self._eye_aspect_ratio(landmarks, self.RIGHT_EYE, w, h)

            avg_ear = (left_ear + right_ear) / 2.0

            # Eye state
            if avg_ear < self.EYE_THRESHOLD:
                self.eye_closed = True
            else:
                self.eye_closed = False

            # Blink detection (state transition)
            if self.prev_eye_closed and not self.eye_closed:
                self.blink_count += 1
                blink_detected = True

            self.prev_eye_closed = self.eye_closed

            # Debug text
            cv2.putText(frame, f"EAR: {avg_ear:.3f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display blink count
        cv2.putText(frame, f"Blinks: {self.blink_count}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        return frame, self.blink_count, blink_detected

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    tracker = EyeTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, count, blinked = tracker.process_frame(frame)

        cv2.imshow("Eye Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
