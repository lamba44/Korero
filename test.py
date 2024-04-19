import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

offset = 22
imgSize = 300
counter = 0

folder = "Data/Y"

cap = cv2.VideoCapture(0)  # Index of Camera being used

detector = HandDetector(maxHands=1)  # Max Number of Hands to detect

classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

labels = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
]

min_confidence = 0.5  # Minimum confidence level for prediction

while True:
    success, img = cap.read()  # Capturing the footage
    imgOutput = img.copy()
    hands, img = detector.findHands(img)  # Detect hand in the footage

    try:
        if hands:  # if hand is detected
            hand = hands[0]  # only one hand to be used, hence index 0
            x, y, w, h = hand["bbox"]  # get value of bounding box dimensions

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[
                y - offset : y + h, x - offset : x + w + 20
            ]  # cropped image size

            imgCropShape = imgCrop.shape

            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)

                imgWhite[:, wGap : wCal + wGap] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)

                imgWhite[hGap : hCal + hGap, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            confidence = prediction[index]  # Confidence value of the prediction

            if confidence >= min_confidence:
                cv2.rectangle(
                    imgOutput,
                    (x - offset, y - offset - 50),
                    (x - offset + 200, y - offset),
                    (255, 0, 255),
                    cv2.FILLED,
                )

                cv2.putText(
                    imgOutput,
                    f"{labels[index]}: {confidence:.2f}",
                    (x, y - 25),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1.75,
                    (255, 255, 255),
                    3,
                )

                cv2.rectangle(
                    imgOutput,
                    (x - offset, y - offset),
                    (x + w + offset, y + h + offset),
                    (255, 0, 255),
                    4,
                )
                cv2.imshow("Image Crop", imgCrop)
                cv2.imshow("Image White", imgWhite)
        else:
            cv2.imshow("Image Crop", imgCrop)  # Show the previous image crop
            cv2.imshow("Image White", imgWhite)  # Show the previous image white
    except:
        pass

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
