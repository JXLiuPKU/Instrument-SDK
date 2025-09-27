"""
需要写四个函数，第一个函数用于打开相机，第二个函数用于初始化，第三个函数用于capture一张图，拍图，第四个函数用于关闭相机；

"""

from pylablib.devices import DCAM
from typing import Optional
import tifffile as tiff
import numpy as np

def open_camera(idx: int = 0) -> DCAM.DCAMCamera:
    """打开第 idx 台 DCAM 相机（0 表示第一台）"""
    camera = DCAM.DCAMCamera(idx=idx)
    return camera

def close_camera(camera: DCAM.DCAMCamera):
    """安全关闭相机；若在采集中会先停止采集。"""
    try:
        if camera.acquisition_in_progress():
            camera.stop_acquisition()
    finally:
        camera.close()

def configure_para(camera: DCAM.DCAMCamera, exposure_s: float,
                   x0: Optional[int] = None, w: Optional[int] = None,
                   y0: Optional[int] = None, h: Optional[int] = None,
                   align_step: int = 4) -> None:
    """设置基础属性：曝光时间（秒）+ 全幅 ROI"""
    # 曝光时间（单位：秒）
    camera.set_attribute_value("EXPOSURE TIME", float(exposure_s))

    det_w, det_h = camera.get_detector_size()  # (width, height)

    # 如果没传ROI的参数，那就默认全幅ROI，ROI需要的参数是起点坐标和终点坐标;
    if None in (x0, y0, w, h):
        hstart, hend = 0, det_w
        vstart, vend = 0, det_h
    else:
        # 这一步是在对齐，因为Hamamatsu的相机要求XYHW四个参数都要被4整除，这一步自动将XYHW转为距离他们自己最近，同时能被4整除的数
        def down(v, s): return int(v // s * s)
        def up(v, s):   return int((v + s - 1) // s * s)

        x0 = down(int(x0), align_step)
        y0 = down(int(y0), align_step)
        w = up(int(w),   align_step)
        h = up(int(h),   align_step)

        hstart, hend = x0, x0 + w
        vstart, vend = y0, y0 + h

    camera.set_roi(hstart, hend, vstart, vend, hbin=1, vbin=1)

    # 回读参数确认范围
    try:
        rhstart, rhend, rvstart, rvend, hbin, vbin = camera.get_roi()
        print(f"[DCAM] ROI set -> H:[{rhstart},{rhend})  V:[{rvstart},{rvend})  "
              f"W={rhend - rhstart}  H={rvend - rvstart}  BIN={hbin}")
    except Exception:
        pass

def snap_one(camera: DCAM.DCAMCamera, save_path: str = "capture.tif"):
    """抓取一帧，并保存为16位 TIFF"""
    frame = camera.snap()               # numpy 2D 数组，通常 dtype=uint16
    tiff.imwrite(save_path, frame)
    print(f"Saved: {save_path}, shape={frame.shape}, dtype={frame.dtype}")
    return frame

def stream_n_frames(camera: DCAM.DCAMCamera,
                    exposure_s: float = 0.01,
                    n_collect: int = 100,
                    ring_buffer: int = 200,
                    per_frame_timeout: float = 2.0) -> list[np.ndarray]:
    """
    连续采集 n_collect 帧并返回（numpy 数组列表）。
    """
    # 设置曝光（属性名是字符串 "EXPOSURE TIME"，单位秒）
    camera.set_attribute_value("EXPOSURE TIME", float(exposure_s))

    # 配置并启动连续采集（sequence）
    camera.setup_acquisition(mode="sequence", nframes=ring_buffer)
    camera.start_acquisition()

    frames = []
    try:
        while len(frames) < n_collect:
            # 等到至少有 1 帧到达（也可设 since="now"/"start"）
            camera.wait_for_frame(nframes=1, timeout=per_frame_timeout)
            new = camera.read_multiple_images()
            if new:
                frames.extend(new)
    finally:
        camera.stop_acquisition()
    return frames[:n_collect]

def snap_background(camera, exposure_s, n_frames):
    """
    50ms 全局 ROI 连续采 n_frames 张，平均得到背景；返回 float32 背景。
    """
    sum_background = None
    for i in range(n_frames):
        frame = camera.snap()
        if sum_background is None:
            sum_background = frame.astype(np.float32)
        else:
            sum_background += frame.astype(np.float32)
    background = sum_background / float(n_frames)

    return background

def main():
    camera = open_camera(idx=0)
    try:
        configure_para(camera, exposure_s=Exposure_s, x0=872, w=612, y0=908, h=612)
        snap_one(camera, "capture.tif")
    finally:
        camera.close()

if __name__ == "__main__":
    main()
