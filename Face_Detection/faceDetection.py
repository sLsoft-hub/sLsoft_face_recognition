import cv2

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
    def detectFaceSmileInVideo( self , videoFrame , cascPath, smile_cascPath, imageShow = 0):
        faceCascade = cv2.CascadeClassifier(cascPath)
        if smile_cascPath is not None:
            smile_cascade = cv2.CascadeClassifier(smile_cascPath)
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
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = videoFrame[y:y + h, x:x + w]
                smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)

        return videoFrame


    # this function accepts video frames and look for face
    def detectFaceEyesInVideo(self, videoFrame, cascPath, eye_cascPath, imageShow=0):
        faceCascade = cv2.CascadeClassifier(cascPath)
        if eye_cascPath is not None:
            eye_cascade = cv2.CascadeClassifier(eye_cascPath)
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
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = videoFrame[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.8, 20)
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


