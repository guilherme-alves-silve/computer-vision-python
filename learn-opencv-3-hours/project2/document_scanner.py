import cv2
from stack_images import stack_images
from document_scanner_utils import *

frame_width = 480
frame_height = 640
brightness = 150

img = cv2.imread('../resources/document.jpg')
img = cv2.resize(img, (frame_width, frame_height))
img_result = img.copy()
img_threshold = pre_processing(img)
biggest_approx = get_contours(img_threshold, img_result, min_threshold=1000, test=True)

if biggest_approx.size > 0:
    img_warped = get_warp(img, biggest_approx, frame_width, frame_height)
    img_stack = stack_images(0.7, ([img, img_threshold], [img_result, img_warped]))
    cv2.imshow("Images", img_stack)
else:
    img_stack = stack_images(0.7, ([img, img], [img_result, img_threshold]))
    cv2.imshow("Images [Not found]", img_stack)

cv2.waitKey(0)
