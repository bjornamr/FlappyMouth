
import cv2
import imutils
import numpy as np
import queue
import dlib
from facialcalc import distanceBetweenMouth, heightFace


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

class MouthDetector:



    def __init__(self, detectorfile):
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(detectorfile)
        self.vid = cv2.VideoCapture(0)
        self.minimumDistances = {}
        self.is_open = False
        self.events = queue.Queue()
        self.id = 0

    def get_last(self):
        try:
            return self.events.get(False)
        except:
            return False

    def detect(self):
        self.id = self.id + 1
        _, frame = self.vid.read()

        # resizing frame
        # you can use cv2.resize but I recommend imutils because its easy to use
        frame = imutils.resize(frame, width=400)

        # grayscale conversion of image because it is computationally efficient
        # to perform operations on single channeled (grayscale) image
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detecting faces
        face_boundaries = self.face_detector(frame_gray, 0)

        for (enum, face) in enumerate(face_boundaries):
            landmarks = self.landmark_predictor(frame_gray, face)
            # converting co-ordinates to NumPy array
            landmarks = land2coords(landmarks)

            key = heightFace(landmarks) // 2
            if key in self.minimumDistances:
                minimumMouthDistance = self.minimumDistances[key]
            elif key-1 in self.minimumDistances:
                minimumMouthDistance = self.minimumDistances[key-1]
            elif key+1 in self.minimumDistances:
                minimumMouthDistance = self.minimumDistances[key+1]
            else:
                minimumMouthDistance = 99999

            distanceMouth = distanceBetweenMouth(landmarks)
            if distanceMouth < minimumMouthDistance:
                self.minimumDistances[key] = distanceMouth

            is_open_now = distanceMouth > 1.30 * minimumMouthDistance

            print(is_open_now)

            if is_open_now != self.is_open:
                self.events.put(is_open_now)

            self.is_open = is_open_now




