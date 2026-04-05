import time
import numpy as np


class AttentionEngine:
    def __init__(self):
        self.blink_history = []
        self.gaze_history = []
        self.emotion_state = "neutral"

        self.last_update = time.time()

    def update(self, blink_count, blink_detected, gaze_score, emotion):
        """
        Inputs:
        - blink_count: total blinks
        - blink_detected: True/False (recent blink event)
        - gaze_score: 0 (distracted) → 1 (focused)
        - emotion: string (neutral, happy, sad, etc.)
        """
        current_time = time.time()

        if blink_detected:
            self.blink_history.append(current_time)

        # keep last 60 seconds
        self.blink_history = [
            t for t in self.blink_history if current_time - t < 60
        ]

        blink_rate = len(self.blink_history)  # blinks per minute

        emotion_score = 1.0

        if emotion == "neutral":
            emotion_score = 1.0
        elif emotion == "happy":
            emotion_score = 1.0
        elif emotion in ["sad", "angry", "fear"]:
            emotion_score = 0.6
        elif emotion == "surprise":
            emotion_score = 0.8

        if blink_rate < 10:
            blink_score = 0.9  # low fatigue
        elif blink_rate < 20:
            blink_score = 0.7
        else:
            blink_score = 0.4  # high fatigue

        gaze_score = np.clip(gaze_score, 0, 1)

        attention_score = (
            (0.4 * gaze_score) +
            (0.3 * blink_score) +
            (0.3 * emotion_score)
        ) * 100

        attention_score = round(attention_score, 2)

        if attention_score > 75:
            status = "Focused"
        elif attention_score > 50:
            status = "Moderate"
        else:
            status = "Distracted"

        return {
            "attention_score": attention_score,
            "status": status,
            "blink_rate": blink_rate,
            "gaze_score": gaze_score,
            "emotion_score": emotion_score
        }
