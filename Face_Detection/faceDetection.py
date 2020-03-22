import cv2
import numpy as np
debug = True
cascPath = "./OpenCV/haarcascades/haarcascade_frontalface_default.xml"
smile_cascPath = "./OpenCV/haarcascades/haarcascade_smile.xml"
eyes_cascPath = "./OpenCV/haarcascades/haarcascade_eye.xml"

GRAY = 1
BGR = 2

class brighnessDetectionClass :

    # This function normalises the brightness of an image to 128
    def changeBrightness( self , videoFrame, imageFormat , brightnessLevel ):
        darkness_threshold = 128
        if imageFormat == GRAY:
            videoFrame_BGR = cv2.cvtColor(videoFrame, cv2.COLOR_GRAY2BGR)
            videoFrame_HSV = cv2.cvtColor(videoFrame_BGR, cv2.COLOR_BGR2HSV)
        else:
            videoFrame_HSV = cv2.cvtColor(videoFrame, cv2.COLOR_BGR2HSV)

        result = cv2.mean( videoFrame_HSV )
        brightnessValue = ( brightnessLevel - result[2] ) + result[2]
        if result[2] > brightnessLevel:
            if debug == True:
                print("bright image = ")
                print(result[2])

        else:
            if debug == True:
                print("dark image = ")
                print(result[2])

        h, s, v = cv2.split(videoFrame_HSV)
        v = cv2.add(v, brightnessValue)
        videoFrame_HSV = cv2.merge((h, s, v))

        if imageFormat == GRAY:
            videoFrame_BGR = cv2.cvtColor(videoFrame_HSV, cv2.COLOR_HSV2BGR)
            videoFrame = cv2.cvtColor(videoFrame_BGR, cv2.COLOR_BGR2GRAY)

        else:
            videoFrame = cv2.cvtColor(videoFrame_HSV, cv2.COLOR_HSV2BGR)

        return videoFrame


class faceDetectionClass:
    def detectFace( self , imagePath , cascPath, imageShow = 0):
        # Read the image
        image = cv2.imread(imagePath)
        faceCascade = cv2.CascadeClassifier(cascPath)

        if image is None:
            print("Image not found.")
            return None
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Detect faces in the image
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            print("Found {0} faces!".format(len(faces)))

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            if imageShow is 1:
                cv2.imshow("Faces found", image)
                cv2.waitKey(0)

            return faces

    # this function accepts video frames and look for face
    def detectFaceSmileInVideo( self , videoFrame ):
        face_detection_scale_factor = 1.3         #scale factor reduces the size of image and passes to detectMultiScale function
        smile_detection_scale_factor = 1.8         #scale factor reduces the size of image and passes to detectMultiScale function
        detectBrightness = brighnessDetectionClass()

        minimumNeighbours = 5

        faceCascade = cv2.CascadeClassifier(cascPath)
        smile_cascade = cv2.CascadeClassifier(smile_cascPath)
        if videoFrame is None:
            print("Image not found.")
            return None
        else:
            gray = cv2.cvtColor(videoFrame, cv2.COLOR_BGR2GRAY)
            gray = detectBrightness.changeBrightness(gray, GRAY, 90)
            videoFrame = detectBrightness.changeBrightness(videoFrame, BGR, 90)
            # Detect faces in the image
            faces = faceCascade.detectMultiScale(
                                                 gray,
                                                 scaleFactor = face_detection_scale_factor,
                                                 minNeighbors = minimumNeighbours ,
                                                 minSize=(30, 30),
                                                 flags=cv2.CASCADE_SCALE_IMAGE
                                                 )
                
            print("Found {0} faces!".format(len(faces)))
                                                
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(videoFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = videoFrame[y:y + h, x:x + w]
                smiles = smile_cascade.detectMultiScale(roi_gray, smile_detection_scale_factor , 20)
                print("Found {0} smiles!".format(len(smiles)))
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)

        return videoFrame


    # this function accepts video frames and look for face
    def detectFaceEyesInVideo(self, videoFrame ):
        detectBrightness = brighnessDetectionClass()
        faceCascade = cv2.CascadeClassifier(cascPath)
        face_detection_scale_factor = 1.2         #scale factor reduces the size of image and passes to detectMultiScale function
        eye_detection_scale_factor = 1.3         #scale factor reduces the size of image and passes to detectMultiScale function
        minimumNeighbours = 5

        eye_cascade = cv2.CascadeClassifier(eyes_cascPath)
        if videoFrame is None:
            print("Image not found.")
            return None
        else:
            gray = cv2.cvtColor(videoFrame, cv2.COLOR_BGR2GRAY)
#            videoFrame = detectBrightness.changeBrightness(videoFrame, BGR, 128)
            gray = detectBrightness.changeBrightness(gray, GRAY , 128 )
            # Detect faces in the image
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor = face_detection_scale_factor,
                minNeighbors = minimumNeighbours,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            print("Found {0} faces!".format(len(faces)))

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(videoFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = videoFrame[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray, eye_detection_scale_factor , minimumNeighbours)
                print("Found {0} eyes!".format(len(eyes)))
                for (sx, sy, sw, sh) in eyes:
                    cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)

        return videoFrame

    # this function accepts video frames and look for face
    def detectFaceInVideo(self, videoFrame, cascPath , imageShow=0):
        faceCascade = cv2.CascadeClassifier(cascPath)
        if videoFrame is None:
            print("Image not found.")
            return None
        else:
            gray = cv2.cvtColor(videoFrame, cv2.COLOR_BGR2GRAY)
            # Detect faces in the image
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            print("Found {0} faces!".format(len(faces)))

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(videoFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return videoFrame


