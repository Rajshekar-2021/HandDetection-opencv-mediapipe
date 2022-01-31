import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm # This is an customized module., and can be used in different projects

cap = cv2.VideoCapture(0)
detector = htm.handDetector()

# Setting the camera window to 640,480 size
wCam, hCam = 640, 480
cap.set(3, wCam) # 3 here indicates the width of the window
cap.set(4, hCam) # 4 indicates the height of the window
pTime = 0        # Initiating the Previous Time to calculate the Frames per second and to display on the window

while True:
    success, img = cap.read()
    img = detector.findHands(img) # Uses the customized module and the internal CLASS : handDetector & method : findHands()
    lmList = detector.findPosition(img) # Uses the customized module and the internal CLASS : handDetector & method : findPosition()
    if len(lmList)!=0:
        print(lmList)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS : {int(fps)}%', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow('Image', img)
    cv2.waitKey(1)
