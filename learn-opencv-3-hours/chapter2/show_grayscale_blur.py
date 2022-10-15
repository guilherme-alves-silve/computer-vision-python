import cv2

img = cv2.imread("../resources/lena.png")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, ksize=(7, 7), sigmaX=0)
cv2.imshow("Gray Image", img_gray)
cv2.imshow("Gray Image Blur", img_blur)
cv2.waitKey(0)
