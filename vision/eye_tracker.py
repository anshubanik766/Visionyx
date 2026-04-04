import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374


class EyeTracker:
    def __init__(self):
        self.gaze_history = []
        self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        gaze = "CENTER"
        confidence = 0
        blink = False

        if result.multi_face_landmarks:
            mesh_points = result.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape

            coords = [(int(p.x * w), int(p.y * h)) for p in mesh_points]

            # Eye positions
            left_eye_left = coords[LEFT_EYE[0]][0]
            left_eye_right = coords[LEFT_EYE[1]][0]
            left_iris_center = sum([coords[i][0] for i in LEFT_IRIS]) // 4

            right_eye_left = coords[RIGHT_EYE[0]][0]
            right_eye_right = coords[RIGHT_EYE[1]][0]
            right_iris_center = sum([coords[i][0] for i in RIGHT_IRIS]) // 4

            if (left_eye_right - left_eye_left) != 0 and (right_eye_right - right_eye_left) != 0:
                left_ratio = (left_iris_center - left_eye_left) / (left_eye_right - left_eye_left)
                right_ratio = (right_iris_center - right_eye_left) / (right_eye_right - right_eye_left)

                avg_ratio = (left_ratio + right_ratio) / 2

                # Smooth
                self.gaze_history.append(avg_ratio)
                if len(self.gaze_history) > 7:
                    self.gaze_history.pop(0)

                smooth_ratio = sum(self.gaze_history) / len(self.gaze_history)

                # Direction
                if smooth_ratio < 0.42:
                    gaze = "LEFT"
                elif smooth_ratio > 0.58:
                    gaze = "RIGHT"
                else:
                    gaze = "CENTER"

                # Confidence
                confidence = round(abs(smooth_ratio - 0.5) * 2, 2)

            # Blink
            left_eye_height = abs(coords[LEFT_EYE_TOP][1] - coords[LEFT_EYE_BOTTOM][1])
            right_eye_height = abs(coords[RIGHT_EYE_TOP][1] - coords[RIGHT_EYE_BOTTOM][1])

            blink = left_eye_height < 5 and right_eye_height < 5

            # Draw iris
            for i in LEFT_IRIS + RIGHT_IRIS:
                cv2.circle(frame, coords[i], 2, (255, 0, 255), -1)

        return frame, gaze, confidence, blink