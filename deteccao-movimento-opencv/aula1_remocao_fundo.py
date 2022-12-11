import cv2
import numpy as np

VIDEO = "Rua.mp4"

cap = cv2.VideoCapture(VIDEO)
has_frame, img = cap.read()

frames_ids = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=72)

frames = []
for fid in frames_ids:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    has_frame, frame = cap.read()
    frames.append(frame)

media_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
cv2.imshow("Media frame", media_frame)
cv2.imwrite("media_frame.jpg", media_frame)
cv2.waitKey(0)
