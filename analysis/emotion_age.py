import cv2
import numpy as np


class EmotionAgeEstimator:
    def __init__(self):
        pass

    def estimate_emotion(self, face_region):

        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)

        if mean_intensity > 140:
            return "Happy"
        elif mean_intensity > 100:
            return "Neutral"
        else:
            return "Sad"

    def estimate_age(self, face_region):
        """
        Rough age estimation based on texture variance.
        (Replace later with ONNX model for accuracy)
        """

        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        variance = np.var(gray)

        if variance < 500:
            return "Child / Teen"
        elif variance < 1200:
            return "Young Adult"
        else:
            return "Adult"

    def analyze(self, frame, bbox):
        """
        Input:
            frame, bbox (x, y, w, h)

        Output:
            emotion, age
        """

        x, y, w, h = bbox
        face_region = frame[y:y+h, x:x+w]

        if face_region.size == 0:
            return "Unknown", "Unknown"

        emotion = self.estimate_emotion(face_region)
        age = self.estimate_age(face_region)

        return emotion, age
