import cv2
import numpy as np

from time import sleep

VIDEO = "Rua.mp4"
delay = 10

cap = cv2.VideoCapture(VIDEO)
has_frame, img = cap.read()

frames_ids = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=72)

frames = []
for fid in frames_ids:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    has_frame, frame = cap.read()
    frames.append(frame)

media_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
cv2.imwrite("media_frame.jpg", media_frame)

#--- Aula 2

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
gray_median_frame = cv2.cvtColor(media_frame, cv2.COLOR_BGR2GRAY)
cv2.imwrite("media_frame_gray.jpg", gray_median_frame)

show_diff = True

while True:
    sleep_time = float(1 / delay)
    sleep(sleep_time)

    has_frame, frame = cap.read()

    if not has_frame:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff_frame = cv2.absdiff(gray_frame, gray_median_frame)
    th, diff_frame = cv2.threshold(diff_frame, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if show_diff:
        cv2.imshow("Diff Frame", diff_frame)
    else:
        cv2.imshow("Frame", gray_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
