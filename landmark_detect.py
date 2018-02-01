import cv2
import imutils
import numpy as np
import dlib
from FlapPyBird.facialcalc import distanceBetweenMouth, heightFace

# http://pythongeeks.net/face-landmark-detection-using-python-and-dlib/
# Function for creating landmark coordinate list
def land2coords(landmarks, dtype="int"):
    # initialize the list of tuples
    # (x, y)-coordinates 19 is mouth
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (a, b)-coordinates
    for i in range(0, 68):  # 19 --- i+48 is mouth
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

    # return the list of (a, b)-coordinates
    return coords
 
# main Function
if __name__=="__main__":
    # loading dlib's Hog Based face detector
    face_detector = dlib.get_frontal_face_detector()
 
    # loading dlib's 68 points-shape-predictor
    # get file:shape_predictor_68_face_landmarks.dat from
    # link: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
 
    # 0 means your default web cam
    vid = cv2.VideoCapture(0)

    minimumDistances = {}

    while True:
        _,frame = vid.read()
 
        # resizing frame
        # you can use cv2.resize but I recommend imutils because its easy to use
        frame = imutils.resize(frame, width=400)
 
        # grayscale conversion of image because it is computationally efficient
        # to perform operations on single channeled (grayscale) image
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
        # detecting faces
        face_boundaries = face_detector(frame_gray, 0)
 
        for (enum,face) in enumerate(face_boundaries):
            # let's first draw a rectangle on the face portion of image
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y

            # Now when we have our ROI(face area) let's
            # predict and draw landmarks
            landmarks = landmark_predictor(frame_gray, face)
            # converting co-ordinates to NumPy array
            landmarks = land2coords(landmarks)

            ratio = int(10*np.sqrt((h / w)))

            key = heightFace(landmarks)//2
            minimumMouthDistance = minimumDistances[key] if key in minimumDistances else 999999

            # Drawing Rectangle on face part
            cv2.rectangle(frame, (x,y), (x+w, y+h), (120, 160, 230), 2)


            distanceMouth = distanceBetweenMouth(landmarks)
            if distanceMouth < minimumMouthDistance:
                minimumDistances[key] = distanceMouth

            print("Distance", distanceMouth, "minimum", minimumMouthDistance, "height", key)
            if distanceMouth > 1.40*minimumMouthDistance:
                print("Your mouth is open")

            for i, (a, b) in enumerate(landmarks):
                # Drawing points on face
                color = (255, 0, 0)
                cv2.circle(frame, (a, b), 2, color, -1)

            # Writing face number on image
            cv2.putText(frame, "Face :{}".format(enum + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 128), 2)
 
        cv2.imshow("frame", frame)
 
        #  Stop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
