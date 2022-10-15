import cv2

img = cv2.imread("../resources/lena.png")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", img_gray)
cv2.waitKey(0)
