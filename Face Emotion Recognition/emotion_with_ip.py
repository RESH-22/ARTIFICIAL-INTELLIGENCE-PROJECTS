from facial_emotion_recognition import EmotionRecognition
import cv2

# Initialize model
er = EmotionRecognition(device='cpu')

# Replace with your phone IP camera address
ip_camera_url = "http://192.168.1.5:8080/video"

cam = cv2.VideoCapture(ip_camera_url)

while True:
    success, frame = cam.read()

    if not success:
        print("IP Camera not connected")
        break

    # Detect emotions
    frame = er.recognize_emotion(frame, return_type='BGR')

    cv2.imshow("Emotion Detection (IP Camera)", frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()