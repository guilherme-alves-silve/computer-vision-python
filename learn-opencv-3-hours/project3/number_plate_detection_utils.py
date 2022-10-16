import cv2


class PlateDetector:

    def __init__(self):
        self._plate_cascade = cv2.CascadeClassifier("../resources/haarcascades/haarcascade_russian_plate_number.xml")

    def detect(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plates = self._plate_cascade.detectMultiScale(img_gray, 1.1, 10)
        return plates


def draw_plates_rect(plates, img_result, min_area=0, color=(255, 0, 255)):
    imgs_roi = []
    for (x, y, w, h) in plates:
        area = w*h
        if area >= min_area:
            cv2.rectangle(img_result, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img_result, "Number plate", (x, y-5),
                       cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
            img_roi = img_result[y:y+h, x:x+w]
            imgs_roi.append(img_roi)

    return imgs_roi
