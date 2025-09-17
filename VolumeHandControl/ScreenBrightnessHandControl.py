from Hand_tracking_module import HandDetector as hd
import math
import cv2
import mediapipe as mp
import time
import screen_brightness_control as sbc

cap = cv2.VideoCapture(0)


p_time = 0
c_time = 0
detector = hd()
while True:
    success, img = cap.read()
    img1 = detector.find_hands(img)
    lm_list = detector.find_position(img1)
    if lm_list:
        x1= lm_list[4][1]
        y1 = lm_list[4][2]

        x2= lm_list[8][1]
        y2 = lm_list[8][2]

        x3 = int((x1+x2)/2)
        y3 = int((y1+y2)/2)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

    
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),5)
        cv2.circle(img,(x1,y1), 7, (0,0,255), -1)
        cv2.circle(img,(x2,y2), 7, (0,0,255), -1)

        cv2.circle(img,(x3,y3), 7, (255,0,255), -1)

        radius =int(math.sqrt((x2-x1)**2 + (y2-y1)**2))

        cv2.circle(img,(447,63), radius, (0,150,255), -1)

        sbc.set_brightness(radius)

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()  
cv2.destroyAllWindows()  
