import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

tip_ids = [4, 8, 12, 16, 20]

def detect_gesture(landmarks):

    fingers = []

    # Thumb
    if landmarks[tip_ids[0]].x > landmarks[tip_ids[0]-1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for i in range(1,5):
        if landmarks[tip_ids[i]].y < landmarks[tip_ids[i]-2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    total = fingers.count(1)

    if total == 0:
        return "FIST"
    elif total == 1:
        return "ONE"
    elif total == 2:
        return "PEACE"
    elif total == 5:
        return "STOP"
    else:
        return "UNKNOWN"


while True:

    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            gesture = detect_gesture(handLms.landmark)

            cv2.putText(img, gesture, (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,255,0), 2)

    cv2.imshow("Hand Gesture Recognition", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()