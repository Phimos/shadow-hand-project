import cv2
import numpy as np
import open3d as o3d


img = cv2.imread("1673601623838.png", cv2.IMREAD_ANYDEPTH)
img = img.astype(np.float32)
cv2.imwrite("1673601623838.png", img)
# print(type(img))
# print(img.dtype)
# print(img.max(), img.min())
# print(img.dtype)
