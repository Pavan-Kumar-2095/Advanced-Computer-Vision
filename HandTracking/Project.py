import cv2
import mediapipe as mp
import time
import Hand_tracking_module as htm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)  # Change to 0 if you want the default camera

detector = htm.HandDetector()

while True:
    success, img = cap.read()
    img = detector.find_hands(img, draw=True)
    lmList = detector.find_position(img, draw=False)
    
    if len(lmList) != 0:
        # Printing the position of the index finger tip (4th landmark)
        print(lmList[4])
    
    # FPS Calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    # Optionally draw FPS on the screen
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)
    
    # Show the image
    cv2.imshow("Hand Tracking", img)
    
    # Exit condition with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


