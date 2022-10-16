import cv2
import numpy as np
from stack_images import stack_images
import color_detection_ui


test_color = None  # [87, 84, 41, 154, 255, 192]
color_detection_ui.init_trackbars(test_color=test_color)

cap = cv2.VideoCapture(0)

while True:

    success, img = cap.read()
    hue_min, hue_max, saturation_min, saturation_max, value_min, value_max \
        = color_detection_ui.get_trackbars_values()

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([hue_min, saturation_min, value_min])
    upper = np.array([hue_max, saturation_max, value_max])
    mask = cv2.inRange(img_hsv, lower, upper)
    img_result = cv2.bitwise_and(img, img, mask=mask)

    img_stacked = stack_images(0.6, ([img, img_hsv], [mask, img_result]))
    cv2.imshow("Images", img_stacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
