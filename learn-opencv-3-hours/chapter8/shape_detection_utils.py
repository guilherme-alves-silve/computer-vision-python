import cv2


def draw_contours(reference_img, img_draw_contours, min_threshold=0):
    # cv2.RETR_EXTERNAL: return the extreme outer contours
    # cv2.CHAIN_APPROX_NONE: get all the contours we have found
    contours, hierarchy = cv2.findContours(reference_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    draw_all_contours = -1
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_threshold:
            cv2.drawContours(img_draw_contours, contour, draw_all_contours,
                             color=(255, 0, 0), thickness=3)
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02*perimeter, True)
            x, y, w, h = cv2.boundingRect(approx)

            object_corners = len(approx)
            object_type: str = "Unknown"
            if object_corners == 3:
                object_type = "Triangle"
            elif object_corners == 4:
                aspect_ratio = w/float(h)
                if 0.98 < aspect_ratio < 1.03:
                    object_type = "Square"
                else:
                    object_type = "Rectangle"
            elif object_corners >= 8:
                object_type = "Circle"

            cv2.rectangle(img_draw_contours, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_draw_contours, object_type,
                        (x+(w//2)-5, y+(h//2)-5), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (0, 0, 0), 2)
