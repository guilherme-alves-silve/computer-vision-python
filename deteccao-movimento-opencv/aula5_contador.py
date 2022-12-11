import cv2
import sys
import numpy as np

from time import sleep

VIDEO = "Ponte.mp4"

background_substractors = dict()

def get_kernel(kernel_type):
    if 'dilation' == kernel_type:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if 'opening' == kernel_type:
        return np.ones((3, 3), np.uint8)
    if 'closing' == kernel_type:
        return np.ones((3, 3), np.uint8)

    print(f"Invalid kernel_type: {kernel_type}\n"
          f"The available kernels are: dilation, opening and closing")
    sys.exit(1)  

def do_filter(img, filter_type):
    if 'dilation' == filter_type:
        return cv2.dilate(img, get_kernel('dilation'), iterations=2)
    if 'opening' == filter_type:
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, get_kernel('opening'), iterations=2)
    if 'closing' == filter_type:
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, get_kernel('closing'), iterations=2)
    if 'combine' == filter_type:
        img_closing = do_filter(img, 'closing')
        img_opening = do_filter(img_closing, 'opening')
        img_dilation = do_filter(img_opening, 'dilation')
        return img_dilation

    print(f"Invalid filter_type: {filter_type}\n"
          f"The available filters are: dilation, opening, closing and combine")
    sys.exit(1)  

def create_background_substractor(algorithm_type: str):
    if 'KNN' == algorithm_type:
        return cv2.createBackgroundSubtractorKNN()
    if 'GMG' == algorithm_type:
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if 'CNT' == algorithm_type:
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    if 'MOG' == algorithm_type:
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if 'MOG2' == algorithm_type:
        return cv2.createBackgroundSubtractorMOG2()

    print(f"Invalid algorithm_type: {algorithm_type}\n"
          f"The available algorithms are: {background_substractors.keys()}")
    sys.exit(1)

def centroide(x, y, w, h):
    x1 = w // 2
    y1 = h // 2
    cx = x + x1
    cy = y + y1
    return cx, cy

for algorithm_type in ['KNN', 'GMG', 'CNT', 'MOG', 'MOG2']:
    background_substractors[algorithm_type] = create_background_substractor(algorithm_type)

w_min = 30 # Largura minima retangulo
h_min = 30 # Largura maxima retangulo
offset = 10 # Erro permitido entre pixel
linha_roi = 500 # Posição da linha de contagem (Region of Interest)
carros = 0

cap = cv2.VideoCapture(VIDEO)
background_subtractor = create_background_substractor(algorithm_type)
begin_tick = cv2.getTickCount()

def main():
    while cap.isOpened:
        has_frame, frame = cap.read()

        if not has_frame:
            break 

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        background_subtractor = background_substractors['GMG']
        mask = background_subtractor.apply(frame)
        mask_filter = do_filter(mask, 'combine')
        cars_after_mask = cv2.bitwise_and(frame, frame, mask=mask_filter)

        cv2.imshow("Frame", frame)
        cv2.imshow("Mask Filter", cars_after_mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end_tick = cv2.getTickCount()
        elapsed_tick = (end_tick - begin_tick) / cv2.getTickFrequency()

main()
