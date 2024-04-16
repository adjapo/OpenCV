import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    min_detection_confidence=0.1,
                    min_tracking_confidence=0.2)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(frame)

    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow('Full Body Tracking', frame)

    if cv2.waitKey(5) & 0xFF == 113:  
        break

cap.release()
cv2.destroyAllWindows()
