from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import winsound
import os

# Alarm settings
frequency = 2500
duration = 1000

def eyeAspectRatio(eye):
    # vertical distances
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # horizontal distance
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear


# Thresholds
earThresh = 0.3
earFrames = 48
count = 0

# Face landmark model
shapePredictor = "shape_predictor_68_face_landmarks.dat"

if not os.path.exists(shapePredictor):
    print("ERROR: shape_predictor_68_face_landmarks.dat not found")
    exit()

# Initialize camera
cam = cv2.VideoCapture(0)

# Initialize detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredictor)

# Eye landmark indexes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:

    ret, frame = cam.read()

    if not ret:
        print("Camera not detected")
        break

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eyeAspectRatio(leftEye)
        rightEAR = eyeAspectRatio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check drowsiness
        if ear < earThresh:
            count += 1

            if count >= earFrames:
                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                winsound.Beep(frequency, duration)

        else:
            count = 0

    cv2.imshow("Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()