import cv2
import numpy as np

img = np.zeros((512, 512, 3), np.uint8)

# black
cv2.imshow("Image Black", img)

# blue
img[:] = 255, 0, 0
cv2.imshow("Image Blue", img)

# green
img[:] = 0, 255, 0
cv2.imshow("Image Green", img)

# red
img[:] = 0, 0, 255
cv2.imshow("Image Red", img)

cv2.waitKey(0)
