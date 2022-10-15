import cv2

img = cv2.imread("../resources/lena.png")

img_blur = cv2.GaussianBlur(img, ksize=(13, 13), sigmaX=0)
cv2.imshow("Image", img)
cv2.imshow("Image Blur", img_blur)
cv2.waitKey(0)
