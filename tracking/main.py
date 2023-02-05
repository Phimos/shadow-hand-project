from typing import List, Tuple

import cv2
import numpy as np

cv2.connectedComponents()

cv2.THRESH_OTSU


def convert_to_binary(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image


def find_connected_components(image: np.ndarray) -> np.ndarray:
    image = convert_to_binary(image)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        image, connectivity=4
    )
    return num_labels, labels, stats, centroids


def detect_circular_landmarks(image: np.ndarray) -> List[Tuple[float, float]]:
    # convert to binary
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # find connected components
    num_labels, labels, stats, centroids = find_connected_components(image)
    # print(f"Found {num_labels} connected components") 
    # print(f"Stats: {stats}")
    # print(f"Centroids: {centroids}")
    # print(f"Labels: {labels}")
