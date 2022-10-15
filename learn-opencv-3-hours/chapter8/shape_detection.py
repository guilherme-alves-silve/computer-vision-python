import numpy as np
from shape_detection_utils import *
from stack_images import stack_images

img = cv2.imread("../resources/shapes.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, ksize=(7, 7), sigmaX=1)
img_canny = cv2.Canny(img_blur, 50, 50)
img_blank = np.zeros_like(img)
img_draw_contours = img.copy()
draw_contours(img_canny, img_draw_contours)

img_stack = stack_images(0.6, ([img, img_gray, img_blur],
                               [img_canny, img_draw_contours, img_blank]))

cv2.imshow("Images", img_stack)
cv2.waitKey(0)
