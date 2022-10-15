import cv2

img = cv2.imread("../resources/lambo.png")
print(img.shape)

# it's height (0:200) then width (200:500)
img_cropped = img[0:200, 200:500]

cv2.imshow("Image", img)
cv2.imshow("Image Cropped", img_cropped)
cv2.waitKey(0)
