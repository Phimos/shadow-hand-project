from typing import Tuple

import cv2
import numpy as np


def error_ellipse_fitting(
    ellipse: Tuple[Tuple[float, float], Tuple[float, float], float], points: np.ndarray
) -> float:
    """Calculates the error of the ellipse fitting.

    The algorithm is based on the following paper:
        `A Buyer's Guide to Conic Fitting` by Fitzgibbon
    section 3.1 - Algorithm LIN: Algebraic Distance Fitting

    Args:
        ellipse (Tuple[Tuple[float, float], Tuple[float, float], float]):
            retval of cv2.fitEllipse
        points (np.ndarray): numpy array of 2D points

    Returns:
        float: error of the ellipse fitting
    """

    points = points.reshape(-1, 2)

    center, size, angle = ellipse
    cx, cy = center
    a, b = size
    a, b = a / 2, b / 2
    theta = np.radians(angle)

    scale = 1 / np.sqrt(a**2 + b**2)
    cx, cy = cx * scale, cy * scale
    a, b = a * scale, b * scale
    points = points * scale

    A = (a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2
    B = 2 * (b**2 - a**2) * np.sin(theta) * np.cos(theta)
    C = (a * np.cos(theta)) ** 2 + (b * np.sin(theta)) ** 2
    D = -2 * A * cx - B * cy
    E = -B * cx - 2 * C * cy
    F = A * cx**2 + B * cx * cy + C * cy**2 - a**2 * b**2

    x, y = points[:, 0], points[:, 1]

    # Equation: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    error = (A * x**2 + B * x * y + C * y**2 + D * x + E * y + F) ** 2
    return error.mean() / (F**2)


if __name__ == "__main__":
    image: np.ndarray = cv2.imread("cup3.jpeg")
    cv2.imshow("Original", image)
    cv2.waitKey(0)

    original = image.copy()

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", image)
    cv2.waitKey(0)

    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 35
    )
    cv2.imshow("Adaptive Threshold", image)
    cv2.waitKey(0)

    image = cv2.Canny(image, 80, 200, apertureSize=3)
    cv2.imshow("Canny", image)
    cv2.waitKey(0)

    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, element)
    cv2.imshow("Morphology", image)
    cv2.waitKey(0)

    find_image = np.zeros_like(image)
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"Found {len(contours)} contours")

    areas = [cv2.contourArea(contour) for contour in contours]
    print(sorted(areas, reverse=True)[:10])

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 1000:
            continue
        cv2.drawContours(find_image, contours, i, 255, 2)

    cv2.imshow("Contours", find_image)
    cv2.waitKey(0)

    # circle_image = original.copy()
    circle_image = find_image.copy()
    contours, hierarchy = cv2.findContours(
        find_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"Found {len(contours)} contours")
    for i in range(len(contours)):
        center, size, angle = cv2.fitEllipse(contours[i])
        error = error_ellipse_fitting((center, size, angle), contours[i])
        center = tuple(map(int, center))
        if error > 1e-8 or max(size) / min(size) > 2:
            continue
        print("Index", i, "Error", error, "Size", size, "Angle", angle)
        circle_image = cv2.drawMarker(
            circle_image, center, (0, 0, 255), cv2.MARKER_CROSS
        )
        cv2.putText(circle_image, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        circle_image = cv2.ellipse(circle_image, (center, size, angle), 255, 2)
    cv2.imshow("Circle", circle_image)
    cv2.waitKey(0)
