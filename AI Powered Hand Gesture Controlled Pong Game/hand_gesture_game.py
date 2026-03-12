import cv2
import mediapipe as mp
import numpy as np
import random
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ----------------------------
# Load Hand Landmarker Model
# ----------------------------
model_path = "hand_landmarker.task"

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

paddle_width = 120
paddle_height = 20
ball_radius = 10

ball_x = random.randint(100, 500)
ball_y = 100
ball_speed_x = 6
ball_speed_y = 6

score = 0
start_time = time.time()

# For smooth motion
previous_paddle_x = 0
smooth_factor = 0.2   # Lower = smoother

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp = int((time.time() - start_time) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp)

    paddle_x = w // 2

    # ----------------------------
    # Finger Tracking
    # ----------------------------
    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]
        index_finger = hand_landmarks[8]

        target_x = int(index_finger.x * w)

        # Smooth Movement Formula
        paddle_x = int(previous_paddle_x +
                       (target_x - previous_paddle_x) * smooth_factor)

        previous_paddle_x = paddle_x
    else:
        paddle_x = previous_paddle_x

    # ----------------------------
    # Ball Movement
    # ----------------------------
    ball_x += ball_speed_x
    ball_y += ball_speed_y

    if ball_x <= ball_radius or ball_x >= w - ball_radius:
        ball_speed_x *= -1

    if ball_y <= ball_radius:
        ball_speed_y *= -1

    if (h - 60 < ball_y + ball_radius < h - 60 + paddle_height) and \
       (paddle_x - paddle_width // 2 < ball_x < paddle_x + paddle_width // 2):
        ball_speed_y *= -1
        score += 1

    if ball_y > h:
        ball_x = random.randint(100, 500)
        ball_y = 100
        score = 0

    # ----------------------------
    # Draw Paddle
    # ----------------------------
    cv2.rectangle(frame,
                  (paddle_x - paddle_width // 2, h - 60),
                  (paddle_x + paddle_width // 2, h - 60 + paddle_height),
                  (255, 0, 0), -1)

    cv2.circle(frame, (ball_x, ball_y), ball_radius, (0, 255, 0), -1)

    cv2.putText(frame, f"Score: {score}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)

    cv2.imshow("AI Hand Gesture Game", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()