import cv2

cascPath = "./OpenCV/haarcascades/haarcascade_frontalface_default.xml"


class faceDetectionClass:

    faceCascade = cv2.CascadeClassifier(cascPath)

    def detectFace( self , imagePath , imageShow = 0):
        # Read the image
        image = cv2.imread(imagePath)
        if image is None:
            print("Image not found.")
            return None
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Detect faces in the image
            faces = self.faceCascade.detectMultiScale(
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


