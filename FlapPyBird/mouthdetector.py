
import cv2
import imutils
import numpy as np
import queue
import dlib
import collections
from facialcalc import distanceBetweenMouth, heightFace
from imutils.video import VideoStream
import time

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

class Mouth:
    def __init__(self):
        self.min_size = collections.deque(maxlen=3)
        self.max_size = collections.deque(maxlen=3)
    
    def empty(self):
        return len(self.min_size)==0

    def get_min(self):
        return np.mean(self.min_size) if not self.empty() else 99999

    def get_max(self):
        return np.mean(self.max_size) if not self.empty() else 0

    def append(self, size, delta = 0):
        size_low = size + delta
        size_high = size - delta
        
        if self.empty():
            self.min_size.append(size_low)
            self.max_size.append(size_high)
            return
        
        mn,mx = self.get_min(), self.get_max()
        avg = 0.5*(mx+mn)

        if size_low < mn:
            self.min_size.append(size_low)
        
        if size_high > mx:
            self.max_size.append(size_high)
        

    def __str__(self):
        return "min: {}, max: {}".format(np.round(self.get_min(),2), np.round(self.get_max(),2) )

class MouthDetector:

    def __init__(self, detectorfile):
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(detectorfile)
        #self.vid = cv2.VideoCapture(0)
        self.vid = VideoStream(usePiCamera=False).start()
        self.mouth_points = np.arange(48,68)
        self.show = False
        self.verbose = False

        self.reset()

    def reset(self):
        self.aspects = {}
        self.last_state = None
        self.tmp_state = None
        self.tmp_state_count = 0
        self.last_time = time.perf_counter()
        self.images = []

    def get_last(self):
        try:
            return self.events.get(False)
        except:
            return None

    def toggle_display(self):
        self.show = not self.show

    def toggle_verbose(self):
        self.verbose = not self.verbose


    def detect_timed(self, resolution=0.00):
        now = time.perf_counter()
        if now - self.last_time >= resolution:
            self.last_time = now
            return self.detect()
        return None
    
    def detect(self, forceImage = False):
        frame = self.vid.read()

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

            distanceMouth = np.round(distanceBetweenMouth(landmarks), 3)

            height = heightFace(landmarks)
            key = height//5

            for k in range(100):
                m = self.aspects.setdefault(k, Mouth())
                m.append(distanceMouth, max(0, k - key )*2)

            mouth = self.aspects[key]
            open_ratio = np.round(distanceMouth / mouth.get_max(),2) if distanceMouth >=3 else 0

            old_state = self.last_state
            is_open = self.update_state( open_ratio )

            if forceImage or (open_ratio > 0.75):
                self.append_image(frame, landmarks[62], distanceMouth, forceImage )

            if self.verbose:
                if is_open:
                    symbol = '[O]' if self.last_state else '[o]'
                else:
                    symbol = '[.]' if self.last_state else '[-]'

                print( symbol , "ratio", open_ratio, mouth, "a", key )

            if self.show:
                
                color = (0, 0, 255) if self.last_state else (255,0,0)
                
                for i in self.mouth_points:
                    cv2.circle(frame, tuple(landmarks[i]), 2, color, -1)

                cv2.imshow("frame", frame)

            if old_state == self.last_state:
                return None
            
            return self.last_state

    def append_image(self,img,mouthXY, mouth_size, force = False):
        
        img_count = len(self.images) 

        max_size = np.max([s for i,s in self.images]) if img_count > 0 else 0

        if not force and (mouth_size < max_size*1.1) and ( (img_count>= 6) or np.random.choice(2) ):
            return

        # print( img.shape )
        w,h=180,222
        
        outX = max(0,mouthXY[0] + w//2 - img.shape[1])
        outY = max(0,mouthXY[1] + h//2 - img.shape[0])

        startX = max(0, mouthXY[0] - w//2 - outX )
        startY = max(0, mouthXY[1] - h//2 - outY )    
        
        img = img[startY:startY+h,startX:startX+w]

        if force:
            self.images.sort(key = lambda x: x[1])

        self.images.append( (img,mouth_size) )
        
    
    def update_state(self, open_ratio):
        
        state = open_ratio >= 0.4

        if open_ratio >= 0.7 or open_ratio <= 0.1:
            repetitions = 0
        elif open_ratio >= 0.6 or open_ratio <= 0.2:
            repetitions = 1
        else:
            repetitions = 3

        if repetitions == 0:
            self.last_state = state
        elif self.tmp_state != state:
            self.tmp_state = state
            self.tmp_state_count = 1
        elif self.tmp_state_count >= repetitions:
            self.tmp_state = None
            self.tmp_state_count = 0
            self.last_state = state
        else:
            self.tmp_state_count += 1 

        return state

            


            




