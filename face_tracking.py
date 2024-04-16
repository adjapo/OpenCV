import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection
face = mp_face.FaceDetection(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face.process(frame)

    if results_face.detections:
        for detection in results_face.detections:
            mp_drawing.draw_detection(frame, detection)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow('Face Tracking', frame)

    if cv2.waitKey(5) & 0xFF == 113:  
        break

cap.release()
cv2.destroyAllWindows()
