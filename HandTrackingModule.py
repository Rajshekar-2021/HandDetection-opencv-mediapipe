import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=4, complexity=1,detectionConf=0.5, trackingConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionConf = detectionConf
        self.trackingConf = trackingConf


        # Initializing the mediapipe solutions with Hands and drawing_utils
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.complexity, self.detectionConf, self.trackingConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # Creating a Mirror Image., and over writing the initial image
        img = cv2.flip(img,1)
        # Converting the above BGR image into RGB Image
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)

        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self,img,handNo=0,draw=True):
        lmList = []
        if self.result.multi_hand_landmarks:
            #myHand = self.result.multi_hand_landmarks[handNo]
            for handLms in self.result.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id,cx,cy])
                    #print(id, cx, cy)
                    if draw:
                        if id==4:
                            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
        return lmList


def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    # Setting the camera window to 640,480 size
    wCam, hCam = 640, 480
    cap.set(3, wCam)
    cap.set(4, hCam)
    pTime = 0

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList)!=0:
            print(lmList)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS : {int(fps)}%', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
