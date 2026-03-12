import cv2
import imutils

cascade_src = "cars/cars.xml"

car_cascade = cv2.CascadeClassifier(cascade_src)

# Use camera 0
cam = cv2.VideoCapture(0)

while True:

    ret, img = cam.read()

    # If camera fails
    if not ret:
        print("Camera not working")
        break

    img = imutils.resize(img, width=1000)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow("Vehicle Detection", img)

    n = len(cars)

    print("-----------------------------")
    print("North:", n)

    if n >= 8:
        print("North More Traffic, Please turn RED Signal")
    else:
        print("No traffic")

    if cv2.waitKey(33) == 27:
        break

cam.release()
cv2.destroyAllWindows()