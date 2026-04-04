import cv2
from deepface import DeepFace


class FaceTracker:
    def __init__(self):
        self.age = None
        self.frame_count = 0

    def get_age(self, frame):
        self.frame_count += 1

        # Run every 30 frames (performance optimization)
        if self.frame_count % 30 == 0:
            try:
                analysis = DeepFace.analyze(
                    frame,
                    actions=['age'],
                    enforce_detection=False
                )
                self.age = analysis[0]['age']
            except:
                pass

        return self.age