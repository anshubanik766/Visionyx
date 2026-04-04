import cv2
from vision.eye_tracker import EyeTracker
from vision.face_tracker import FaceTracker


def run():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not accessible")
        return

    eye_tracker = EyeTracker()
    face_tracker = FaceTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, gaze, confidence, blink = eye_tracker.process(frame)
        age = face_tracker.get_age(frame)

        # Display
        cv2.putText(frame, f"Gaze: {gaze}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Confidence: {confidence}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if blink:
            cv2.putText(frame, "BLINK", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if age is not None:
            cv2.putText(frame, f"Age: {int(age)}", (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), 2)

        cv2.imshow("Visionyx", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()