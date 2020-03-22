from Face_Detection import faceDetection
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
fd = faceDetection.faceDetectionClass()

if 0:
    image = "./Images/Img_01.JPG"
    face = fd.detectFace(image,cascPath)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True :
        # Display the resulting frame
        face = fd.detectFaceEyesInVideo(frame)
#        face = fd.detectFaceSmileInVideo(frame)
        cv2.imshow('frame',face)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
