import cv2
import sys
import numpy as np

from time import sleep

ESC = 27
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

w_min = 40 # Largura minima retangulo
h_min = 40 # Largura maxima retangulo
offset = 2 # Erro permitido entre pixel
linha_roi = 620 # Posição da linha de contagem (Region of Interest)
carros = 0

def set_info(detected, frame):
    global carros
    for x, y in detected:
        if (linha_roi + offset) > y > (linha_roi - offset):
            carros += 1
            cv2.line(frame, (25, linha_roi), (1200, linha_roi), (0, 127, 255), 3)
            detected.remove((x, y))
            print(f"Carros detectados até o momento: {carros}")

def show_info(frame, mask, show_mask=False):
    text = f'Carros: {carros}'
    cv2.putText(frame, text, (200, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Original", frame)    
    if show_mask:
        cv2.imshow("Detectado", mask)

for algorithm_type in ['KNN', 'GMG', 'CNT', 'MOG', 'MOG2']:
    background_substractors[algorithm_type] = create_background_substractor(algorithm_type)

cap = cv2.VideoCapture(VIDEO)
background_subtractor = create_background_substractor(algorithm_type)
detected = []
begin_tick = cv2.getTickCount()

def main():
    while cap.isOpened:
        has_frame, frame = cap.read()

        if not has_frame:
            break 

        background_subtractor = background_substractors['GMG']
        mask = background_subtractor.apply(frame)
        mask = do_filter(mask, 'combine')

        contorno, img = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.line(frame, (25, linha_roi), (1200, linha_roi), (255, 127, 0), 3)        
        for i, c in enumerate(contorno):
            x, y, w, h = cv2.boundingRect(c)
            contorno_valido = (w >= w_min) and (h >= h_min)
            if not contorno_valido:
                continue

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            centro = centroide(x, y, w, h)
            detected.append(centro)
            cv2.circle(frame, centro, 4, (0, 0, 255), -1)
        
        set_info(detected, frame)
        show_info(frame, mask)

        if cv2.waitKey(1) == ESC:
            break

        end_tick = cv2.getTickCount()
        elapsed_tick = (end_tick - begin_tick) / cv2.getTickFrequency()

    cap.release()
    cv2.destroyAllWindows()

main()
