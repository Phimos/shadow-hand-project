import time

import pyrealsense2 as rs


if __name__ == "__main__":
    width = 1280
    height = 720
    fps = 6

    ctx = rs.context()
    devices = ctx.query_devices()
    n_camera = len(devices)

    print(f"Number of cameras found: {n_camera}")
    print(f"Width: {width}, Height: {height}")

    for i, device in enumerate(devices):
        print("-" * 30)
        print(f"Index: {i}")
        serial = device.get_info(rs.camera_info.serial_number)
        print(f"Serial number: {serial}")

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        profile = pipeline.start(config)

        color_profile = profile.get_stream(rs.stream.color)
        intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

        print(f"Intrinsics:")
        print(f"\tfx:  {intrinsics.fx}")
        print(f"\tfy:  {intrinsics.fy}")
        print(f"\tppx: {intrinsics.ppx}")
        print(f"\tppy: {intrinsics.ppy}")

        pipeline.stop()

        time.sleep(3)
