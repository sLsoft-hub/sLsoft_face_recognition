from Face_Detection import faceDetection
import cv2

image = "./Images/Img_01.JPG"

fd = faceDetection.faceDetectionClass()
face = fd.detectFace(image,1)

if face is None:
    print("Not cool.")
#else:
#    cv2.imshow("Faces found", face)
#    cv2.waitKey(0)
