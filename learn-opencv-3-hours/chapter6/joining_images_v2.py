import cv2
from stack_images import stack_images

img = cv2.imread("../resources/lena.png")
img_horizontal_stack = stack_images(0.5, ([img, img, img]))
img_vertical_stack = stack_images(0.5, ([img], [img], [img]))
img_horizontal_and_vertical_stack = stack_images(0.5, ([img, img, img], [img, img, img], [img, img, img]))

cv2.imshow("Horizontal Stack", img_horizontal_stack)
cv2.imshow("Vertical Stack", img_vertical_stack)
cv2.imshow("Horizontal and Vertical Stack", img_horizontal_and_vertical_stack)
cv2.waitKey(0)
