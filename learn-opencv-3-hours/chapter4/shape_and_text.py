import cv2
import numpy as np

img = np.zeros((512, 512, 3), np.uint8)

cv2.line(img, pt1=(img.shape[1], img.shape[0]), pt2=(0, 0), color=(0, 0, 255), thickness=3)
cv2.line(img, pt1=(0, 0), pt2=(300, 300), color=(0, 255, 0), thickness=3)
cv2.rectangle(img, (0, 0), (250, 350), (0, 0, 255), cv2.FILLED)
cv2.rectangle(img, (0, 0), (250, 350), (0, 255, 0), 2)
cv2.circle(img, (400, 50), 30, (255, 255, 0), 5)

cv2.putText(img, "OPENCV", (300, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 1)

cv2.imshow("Image", img)
cv2.waitKey(0)
