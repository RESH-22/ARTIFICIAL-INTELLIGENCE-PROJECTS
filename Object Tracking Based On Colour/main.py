import cv2
import imutils
import numpy as np

# HSV range for RED color
lower_red1 = (0, 120, 70)
upper_red1 = (10, 255, 255)

lower_red2 = (170, 120, 70)
upper_red2 = (180, 255, 255)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera not detected")
        break

    # resize frame
    frame = imutils.resize(frame, width=800)

    # blur to remove noise
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    # convert to HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # mask for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 + mask2

    # remove noise
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=3)
    mask = cv2.GaussianBlur(mask, (9,9), 0)

    # find contours
    cnts,_ = cv2.findContours(mask.copy(),
                              cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) > 0:

        # largest contour
        c = max(cnts, key=cv2.contourArea)

        ((x,y), radius) = cv2.minEnclosingCircle(c)

        M = cv2.moments(c)

        if M["m00"] > 0:

            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

            if radius > 40:

                # draw circle
                cv2.circle(frame,
                           (int(x),int(y)),
                           int(radius),
                           (0,255,255),
                           2)

                # draw center
                cv2.circle(frame,
                           center,
                           5,
                           (0,0,255),
                           -1)

                cv2.putText(frame,
                            "Red Box Detected",
                            (center[0]-50, center[1]-20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0,0,255),
                            2)

    cv2.imshow("Tracking", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()