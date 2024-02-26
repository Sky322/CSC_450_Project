import cv2
import mediapipe as mp


def main():
    cap = cv2.VideoCapture(0)  # Open the default camera
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break

            # Convert the image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe
            results = pose.process(frame_rgb)

            # Draw landmarks on the face
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                          mp_pose.POSE_CONNECTIONS)

            # Display the resulting frame
            cv2.imshow('Gaze Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
0.

