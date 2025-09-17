import cv2
import mediapipe as mp
import time

class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        # Ensure staticMode is boolean (True or False)
        if not isinstance(staticMode, bool):
            raise ValueError("staticMode should be a boolean (True or False).")
        
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        
        # Create FaceMesh object with correct parameters (staticMode should be True or False)
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode, 
            max_num_faces=self.maxFaces, 
            min_detection_confidence=self.minDetectionCon, 
            min_tracking_confidence=self.minTrackCon
        )
        
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])  # Append each landmark's coordinates to the face list
                faces.append(face)  # Append landmarks of the detected face
        
        return img, faces

def main():
    cap = cv2.VideoCapture("Face.mp4")  # Replace with your video path
    pTime = 0
    detector = FaceMeshDetector(staticMode=False, maxFaces=2)  # Adjust maxFaces to the desired number of faces to detect
    
    while True:
        success, img = cap.read()
        img= cv2.resize(img, (640, 480))  # Resize image to smaller size
        if not success:
            break  # Exit the loop if the video ends
        
        img, faces = detector.findFaceMesh(img)
        
        if len(faces) != 0:
            print(faces[0])  # Print the landmarks of the first face (if detected)
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        
        cv2.imshow("Image", img)  # Display the processed image
        cv2.waitKey(1)  # Wait for 1 ms to process key events (useful for quitting with a key)

    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    main()

