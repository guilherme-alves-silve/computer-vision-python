import cv2

img = cv2.imread("../resources/lambo.png")
print(img.shape)

img_resized = cv2.resize(img, dsize=(300, 200))
print(img_resized.shape)

cv2.imshow("Image", img)
cv2.imshow("Image Resized", img_resized)
cv2.waitKey(0)
