import cv2
import numpy as np

img = cv2.imread("../resources/lena.png")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, ksize=(7, 7), sigmaX=0)
img_canny = cv2.Canny(img_gray, 150, 150)

kernel = np.ones((5, 5), np.uint8)
img_dilation = cv2.dilate(img_canny, kernel, iterations=1)
img_eroded = cv2.erode(img_dilation, kernel, iterations=1)

cv2.imshow("Canny Image", img_canny)
cv2.imshow("Dilation Image", img_dilation)
cv2.imshow("Eroded Image", img_eroded)
cv2.waitKey(0)
