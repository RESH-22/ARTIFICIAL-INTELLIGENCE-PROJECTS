import cv2
import time
import imutils

cam = cv2.VideoCapture(0)
time.sleep(2)

area = 3000  # Increased threshold for precision

# Use Background Subtractor (Much better than firstFrame method)
backSub = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=False
)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=800)

    # Apply background subtraction
    fgMask = backSub.apply(frame)

    # Remove noise
    fgMask = cv2.GaussianBlur(fgMask, (5, 5), 0)
    _, thresh = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)

    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=3)

    contours, _ = cv2.findContours(thresh,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    text = "Normal"

    for c in contours:
        if cv2.contourArea(c) < area:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)
        text = "Moving Object Detected"

    cv2.putText(frame, text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)

    cv2.imshow("Camera Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()