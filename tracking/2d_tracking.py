import pyrealsense2 as rs
import numpy as np
import time
import cv2
import sys
import os

class Track:
    @classmethod
    def error_ellipse_fitting(cls, ellipse : tuple [tuple[float, float], tuple[float, float], float], points: np.ndarray
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

    @classmethod
    def visualize(cls, file : str, type: int, save:tuple):
        """visualize a marcker-tracking process 

        Args:
            file (str): a picture or a address of the picture
            type (int): the type of file
            save (tuple): save the specular process result
        """        

        ## origin
        if type==0:
            image: np.ndarray = cv2.imread(file)
        else:
            image: np.ndarray = np.array(file)
        print(image.shape)
        cv2.imshow("Original", image)
        cv2.waitKey(0)

        original = image.copy()
        ## Gray
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Gray", image)
        cv2.waitKey(0)

        ## adaptive Threshold
        image = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 15
        )
        cv2.imshow("Adaptive Threshold", image)
        cv2.waitKey(0)
        if 3 in save:
            cv2.imwrite('{}_{}{}'.format(str(time.time())[-4:],'3','.jpeg'),image)


        image = cv2.Canny(image, 80, 200, apertureSize=3)
        cv2.imshow("Canny", image)
        cv2.waitKey(0)
        if 4 in save:
            cv2.imwrite('{}_{}{}'.format(str(time.time())[-4:],'4','.jpeg'),image)

        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, element)
        cv2.imshow("Morphology", image)
        cv2.waitKey(0)
        if 5 in save:
            cv2.imwrite('{}_{}{}'.format(str(time.time())[-4:],'5','.jpeg'),image)

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
        if 6 in save:
            cv2.imwrite('{}_{}{}'.format(str(time.time())[-4:],'6','.jpeg'),find_image)

        # circle_image = original.copy()
        circle_image = find_image.copy()
        contours, hierarchy = cv2.findContours(
            find_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        print(f"Found {len(contours)} contours")
        for i in range(len(contours)):
            center, size, angle = cv2.fitEllipse(contours[i])
            error = cls.error_ellipse_fitting((center, size, angle), contours[i])
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
        if 7 in save:
            cv2.imwrite('{}_{}{}'.format(str(time.time())[-4:],'7','.jpeg'),circle_image)

def shoot(save:int)-> np.ndarray:
    """shoot an image

    Args:
        save (int): whether save the image

    Returns:
        np.ndarray: the image
    """    
    ctx = rs.context()
    if len(ctx.devices) == 0:
        print("No realsense D415 was detected.")
        exit()
    device = ctx.devices[1]
    serial_number = device.get_info(rs.camera_info.serial_number)
    config = rs.config()
    config.enable_device(serial_number)

    # step2 根据配置文件设置数据流
    config.enable_stream(rs.stream.color,
                         1920, 1080,
                         rs.format.bgr8, 30)

    # step3 启动相机流水线并设置是否自动曝光
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    color_sensor = pipeline.get_active_profile().get_device().query_sensors()[1]  # 0-depth(两个infra)相机, 1-rgb相机,2-IMU
    # 自动曝光设置
    color_sensor.set_option(rs.option.enable_auto_exposure, True)


    print("Shooting in camera one")
    # while 1:
    #     frame = pipeline.wait_for_frames()
    #     frame_data = np.asanyarray(frame.get_color_frame().get_data())
    #     timestamp_ms = frame.timestamp

    #     cv2.imshow("received frames", frame_data)
    #     cv2.waitKey(int(1))
    output_type_image='.jpeg'
    time.sleep(3)
    frame=pipeline.wait_for_frames()
    timestamp_ms = frame.timestamp
    output_dir_image='./tracking'
    frame_data = np.asanyarray(frame.get_color_frame().get_data())
    cv2.imwrite(output_dir_image + "/" + str(timestamp_ms) + output_type_image, frame_data)
    return np.array(frame_data)

if __name__ == '__main__':
    Track.visualize('1678111200476.8838.jpeg',0,(3,7))