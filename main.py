from Face_Detection import faceDetection
import cv2

image = "./Images/Img_01.JPG"
cascPath = "./OpenCV/haarcascades/haarcascade_frontalface_default.xml"

fd = faceDetection.faceDetectionClass()
face = fd.detectFace(image,cascPath)

if face is None:
    print("Not cool.")
