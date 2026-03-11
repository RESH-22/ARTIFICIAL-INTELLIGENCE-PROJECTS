import cv2

ip_camera_url = "http://192.168.1.5:8080/video"

cam = cv2.VideoCapture(ip_camera_url)

while True:
    success, frame = cam.read()

    if not success:
        break

    cv2.imshow("IP Camera", frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()