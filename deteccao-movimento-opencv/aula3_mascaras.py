import cv2
import sys
import csv
import numpy as np

from time import sleep

VIDEO = "Ponte.mp4"

background_substractors = dict()

def create_background_substractor(algorithm_type: str):
    if algorithm_type == 'KNN':
        return cv2.createBackgroundSubtractorKNN()
    if algorithm_type == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if algorithm_type == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    if algorithm_type == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if algorithm_type == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2()
    print(f"Invalid algorithm_type: {algorithm_type}\n"
          f"The available algorithms are: {background_substractors.keys()}")
    sys.exit(1)

for algorithm_type in ['KNN', 'GMG', 'CNT', 'MOG', 'MOG2']:
    background_substractors[algorithm_type] = create_background_substractor(algorithm_type)

delay = 10

cap = cv2.VideoCapture(VIDEO)
background_subtractor = create_background_substractor(algorithm_type)
begin_tick = cv2.getTickCount()

def main():
    with open('results.csv', mode='w') as fp:
        writer = csv.DictWriter(fp, fieldnames=['Frame', 'Pixel Count'])
        writer.writeheader()
        while cap.isOpened:
            has_frame, frame = cap.read()

            if not has_frame:
                break 

            frame = cv2.resize(frame, (0, 0), fx=0.35, fy=0.35)

            cv2.imshow("Frame", frame)
            for algorithm, background_subtractor in background_substractors.items():
                mask = background_subtractor.apply(frame)
                algorithm_count = np.count_nonzero(mask)
                writer.writerow({'Frame': algorithm, 'Pixel Count': algorithm_count})
                cv2.imshow(algorithm, mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            end_tick = cv2.getTickCount()
            elapsed_tick = (end_tick - begin_tick) / cv2.getTickFrequency()

main()
