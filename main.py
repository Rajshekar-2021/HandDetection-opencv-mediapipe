import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
detector = htm.handDetector()

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
