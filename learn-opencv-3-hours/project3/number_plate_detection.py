import cv2
from stack_images import stack_images
from number_plate_detection_utils import *

frame_width = 480
frame_height = 640
brightness = 150

imgs_path = [
    '../resources/car-1.jpg',
    '../resources/car-2.jpg',
    '../resources/car-3.jpg',
    '../resources/car-4.jpg'
]

plate_detector = PlateDetector()

processed_imgs = []
imgs_rois = []
for img_path in imgs_path:
    img = cv2.imread(img_path)
    plates = plate_detector.detect(img)
    imgs_rois.extend(draw_plates_rect(plates, img))
    processed_imgs.append(img)

img_stack = stack_images(0.7, (processed_imgs[0:2], processed_imgs[2:]))

if imgs_rois:
    img_roi_stack = stack_images(0.7, (imgs_rois,))
    cv2.imshow("Images ROI", img_roi_stack)

cv2.imshow("Images", img_stack)
cv2.waitKey(0)
