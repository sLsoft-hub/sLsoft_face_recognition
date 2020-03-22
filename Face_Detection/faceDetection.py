import cv2
import numpy as np
debug = True
cascPath = "./OpenCV/haarcascades/haarcascade_frontalface_default.xml"
smile_cascPath = "./OpenCV/haarcascades/haarcascade_smile.xml"
eyes_cascPath = "./OpenCV/haarcascades/haarcascade_eye.xml"

class brighnessDetectionClass :

    def changeBrightness( self , videoFrame ):
        darkness_threshold = 128
        alpha = 2.2
        beta = 50
        videoFrame_HSV = cv2.cvtColor(videoFrame, cv2.COLOR_BGR2HSV)
        result = cv2.mean( videoFrame_HSV )
        if result[2] > darkness_threshold:
            if debug == True:
                print("bright image = ")
                print(result[2])

            new_videoFrame = np.zeros(videoFrame.shape, videoFrame.dtype)
            for y in range(videoFrame.shape[0]):
                for x in range(videoFrame.shape[1]):
                    for c in range(videoFrame.shape[2]):
                        new_videoFrame[y, x, c] = np.clip(alpha * videoFrame[y, x, c] + beta, 0, 255)

        else:
            if debug == True:
                print("dark image = ")
                print(result[2])

            new_videoFrame = np.zeros(videoFrame.shape, videoFrame.dtype)
            for y in range(videoFrame.shape[0]):
                for x in range(videoFrame.shape[1]):
                    for c in range(videoFrame.shape[2]):
                        new_videoFrame[y, x, c] = np.clip(alpha * videoFrame[y, x, c] + beta, 0, 255)

        videoFrame_BGR = cv2.cvtColor(videoFrame, cv2.COLOR_HSV2BGR)
        return videoFrame_BGR


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

        minimumNeighbours = 5

        faceCascade = cv2.CascadeClassifier(cascPath)
        smile_cascade = cv2.CascadeClassifier(smile_cascPath)
        if videoFrame is None:
            print("Image not found.")
            return None
        else:
            gray = cv2.cvtColor(videoFrame, cv2.COLOR_BGR2GRAY)
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
            videoFrame = detectBrightness.changeBrightness(videoFrame)
            gray = cv2.cvtColor(videoFrame, cv2.COLOR_BGR2GRAY)
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


