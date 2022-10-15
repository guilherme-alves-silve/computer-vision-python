import cv2

print(f"Package imported in version {cv2.__version__}")

img = cv2.imread("../resources/lena.png")
cv2.imshow("Output", img)
cv2.waitKey(0)
