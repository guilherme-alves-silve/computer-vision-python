import cv2

cap = cv2.VideoCapture("../resources/test_video.mp4")

frame_width = 640
frame_height = 480

while True:
    success, img = cap.read()
    img = cv2.resize(img, (frame_width, frame_height))
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
