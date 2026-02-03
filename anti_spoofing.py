import cv2
import mediapipe as mp
import numpy as np
import time
import os

model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

detector = FaceLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

blink_detected = False
turn_detected = False
is_authenticated = False

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect(mp_image)

    if results.face_landmarks:
        face = results.face_landmarks[0]

        eye_dist = abs(face[159].y - face[145].y)
        if eye_dist < 0.007: 

            blink_detected = True

        nose_x = face[1].x
        if nose_x < 0.4 or nose_x > 0.6: 

            turn_detected = True

    if blink_detected and turn_detected:
        is_authenticated = True

    color = (0, 255, 0) if is_authenticated else (0, 0, 255)
    status = "IDENTITY VERIFIED" if is_authenticated else "LIVENESS CHECK REQUIRED"

    cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if not is_authenticated:
        cv2.putText(frame, f"Blink: {'[OK]' if blink_detected else '[WAITING]'}", (30, 90), 0, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Turn Head: {'[OK]' if turn_detected else '[WAITING]'}", (30, 120), 0, 0.6, (255, 255, 255), 1)

    cv2.imshow("Anti-Spoofing Auth", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

detector.close()
cap.release()
cv2.destroyAllWindows()