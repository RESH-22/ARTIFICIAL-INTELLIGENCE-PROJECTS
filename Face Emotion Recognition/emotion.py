from facial_emotion_recognition import EmotionRecognition
import cv2

# Initialize emotion recognition model
er = EmotionRecognition(device='cpu')

# Start webcam
cam = cv2.VideoCapture(0)

while True:
    success, frame = cam.read()

    if not success:
        print("Camera not working")
        break

    # Detect emotion
    frame = er.recognise_emotion(frame, return_type='BGR')

    cv2.imshow("Emotion Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()