import cv2

img = cv2.imread("image.png", cv2.IMREAD_UNCHANGED)

print(img.shape)
print(img.dtype)