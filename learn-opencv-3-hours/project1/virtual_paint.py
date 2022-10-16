import cv2
from virtual_paint_utils import *

frame_width = 640
frame_height = 480
brightness = 150
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(10, brightness)

filter_points = []
while True:
    success, img = cap.read()
    img_result = img.copy()
    new_points = find_color(img, img_result)

    if new_points:
        for point in new_points:
            filter_points.append(point)

    if filter_points:
        draw_on_canvas(filter_points, img_result)

    cv2.imshow("Result", img_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
