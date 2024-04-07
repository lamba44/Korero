import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

offset = 20
imgSize = 300
counter = 0

folder = "Data/Y"

cap = cv2.VideoCapture(0)  # Index of Camera being used

detector = HandDetector(maxHands=1)  # Max Number of Hands to detect

while True:
    success, img = cap.read()  # Capturing the footage
    hands, img = detector.findHands(img)  # Detect hand in the footage

    try:
        if hands:  # if hand is detected
            hand = hands[0]  # only one hand to be used, hence index 0
            x, y, w, h = hand["bbox"]  # get value of bounding box dimensions

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset : y + h, x - offset : x + w]  # cropped image size

            imgCropShape = imgCrop.shape

            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)

                imgWhite[:, wGap : wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)

                imgWhite[hGap : hCal + hGap, :] = imgResize

            cv2.imshow("Image Crop", imgCrop)
            cv2.imshow("Image White", imgWhite)
        else:
            cv2.imshow("Image Crop", imgCrop)  # Show the previous image crop
            cv2.imshow("Image White", imgWhite)  # Show the previous image white
    except:
        pass

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imgWhite)
        print(counter)
