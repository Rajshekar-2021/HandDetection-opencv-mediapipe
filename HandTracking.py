# Importing necessary libraries
import cv2
import mediapipe as mp
import time

#Capture the video from the web cam
cap = cv2.VideoCapture(0)

# Setting the camera window to 640,480 size
wCam,hCam = 640,480
cap.set(3,wCam)
cap.set(4,hCam)

# Initializing the mdeispipe solutions with Hands and drawing_utils
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

#Initializing the time, to calculate the Frames per Second (FPS)
pTime = 0

while True:
    success,img = cap.read()

    # The initial Capture will not create the Mirror Image and hence need to flip the image to create Mirror Image
    #Creating a Mirror Image., and overwritting the initial image
    img = cv2.flip(img,1)

    # The Class mediapipe works with RGB image, while the opencv initially the captures the video in BGR
    #Converting the above BGR image into RGB Image
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    # Using mediapipe.solutions.hands.Hands()., we need to get the initial co-ordinates in a variable 'result'
    result = hands.process(imgRGB)
    #print(result.multi_hand_landmarks)
    
    #Checking the results and if there is any Hand detected the loop starts to draw hand landmarks
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                #print(id,lm)
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                # Identifying the Thumb, and marking with a big circle.
                if id==4:
                    cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime


    cv2.putText(img,f'FPS : {int(fps)}%',(20,70),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
    cv2.imshow('Image',img)
    cv2.waitKey(1)





