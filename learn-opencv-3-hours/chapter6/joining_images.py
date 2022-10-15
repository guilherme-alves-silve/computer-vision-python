import cv2
import numpy as np

img = cv2.imread("../resources/lena.png")

img_horizontal = np.hstack((img, img))
img_vertical = np.vstack((img, img))

cv2.imshow("Image Horizontal", img_horizontal)
cv2.imshow("Image Vertical", img_vertical)
cv2.waitKey(0)
