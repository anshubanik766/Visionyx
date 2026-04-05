import cv2
import numpy as np

from vision.face_tracker import FaceTracker
from vision.eye_tracker import EyeTracker
from analysis.emotion_age import EmotionAgeEstimator


class Visionyx:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        self.face_tracker = FaceTracker()
        self.eye_tracker = EyeTracker()
        self.emotion_engine = EmotionAgeEstimator()

        self.frame_id = 0

    def get_faces(self, frame):
        if hasattr(self.face_tracker, "detect"):
            return self.face_tracker.detect(frame)
        if hasattr(self.face_tracker, "get_faces"):
            return self.face_tracker.get_faces(frame)
        if hasattr(self.face_tracker, "track"):
            return self.face_tracker.track(frame)
        if hasattr(self.face_tracker, "process"):
            return self.face_tracker.process(frame)
        return []

    def draw_panel(self, frame, lines):
        x1, y1 = 15, 15
        width = 340
        line_h = 28
        height = 25 + len(lines) * line_h
        x2, y2 = x1 + width, y1 + height

        overlay = frame.copy()

        # glass background
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # glow border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # inner highlight
        cv2.rectangle(frame, (x1+2, y1+2), (x2-2, y2-2), (255, 255, 255), 1)

        # header
        cv2.rectangle(frame, (x1, y1), (x2, y1 + 30), (0, 255, 255), -1)

        cv2.putText(
            frame,
            "VISIONYX HUD",
            (x1 + 10, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

        # text
        for i, text in enumerate(lines):
            y = y1 + 55 + i * line_h
            cv2.putText(
                frame,
                text,
                (x1 + 12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

    def draw_face_mesh(self, frame, bbox):
        x, y, w, h = bbox

        points = [
            (x, y),
            (x + w, y),
            (x, y + h),
            (x + w, y + h),
            (x + w // 2, y),
            (x + w // 2, y + h),
            (x, y + h // 2),
            (x + w, y + h // 2),
            (x + w // 2, y + h // 2),
        ]

        # dots
        for px, py in points:
            cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

        # mesh connections
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 3),
            (4, 5),
            (6, 7),
            (0, 8), (1, 8), (2, 8), (3, 8)
        ]

        for i, j in connections:
            cv2.line(frame, points[i], points[j], (0, 140, 255), 1)

    def draw_focus_ring(self, frame, bbox):
        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2
        r = max(w, h) // 2

        cv2.circle(frame, (cx, cy), r + 10, (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), r - 10, (255, 0, 0), 2)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_id += 1

            faces = self.get_faces(frame)

            for face in faces:
                bbox = face["bbox"]
                x, y, w, h = bbox

                # face box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # UI overlays
                self.draw_focus_ring(frame, bbox)
                self.draw_face_mesh(frame, bbox)

                # emotion + age (safe call)
                try:
                    emotion, age_group = self.emotion_engine.analyze(frame, bbox)
                except:
                    emotion, age_group = "unknown", "unknown"

                # eyes (safe fallback)
                try:
                    eyes = self.eye_tracker.detect(frame, bbox)
                    for ex, ey in eyes:
                        cv2.circle(frame, (ex, ey), 4, (0, 0, 255), -1)
                except:
                    pass

                # HUD panel
                self.draw_panel(frame, [
                    f"Emotion: {emotion}",
                    f"Age: {age_group}",
                    f"Faces: {len(faces)}",
                    f"Frame: {self.frame_id}"
                ])

            cv2.imshow("Visionyx", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = Visionyx()
    app.run()
