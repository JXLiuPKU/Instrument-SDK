from pathlib import Path
import time
import cv2

from DahengCamera_old import CameraQPD  # 按你的命名使用

# 固定测试参数（可按需改常量）
CAMERA_ID   = 1
X_START     = 0
Y_START     = 0
X_WIDTH     = 304
Y_WIDTH     = 304
EXPOSURE_US = 1200.0
GAIN        = 0.0
FRAMES      = 5
SAVE_DIR    = "./cam_selftest_out"
OFFSET_FILE = "./cam_selftest_offset.txt"   # 不写入你原有文件，避免污染

def main():
    # 准备保存目录与 offset 文件（出错就直接抛）
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    Path(OFFSET_FILE).write_text(f"{X_START},{Y_START}", encoding="utf-8")

    # 打开相机（你的 CameraQPD 内部会 setAOI 并尝试开流）
    cam = CameraQPD(camera_id=CAMERA_ID,
                    x_width=X_WIDTH, y_width=Y_WIDTH,
                    sigma=8.0, offset_file=OFFSET_FILE, background=0)

    # 直接设底层曝光/增益
    dev = cam.cam.cam
    dev.ExposureTime.set(float(EXPOSURE_US))
    dev.Gain.set(float(GAIN))

    # 强制关/设 AOI/开流一遍
    cam.cam.stream_off()
    cam.cam.setAOI(int(X_START), int(Y_START), int(X_WIDTH), int(Y_WIDTH))
    cam.cam.stream_on()

    for i in range(FRAMES):
        img = cam.capture()
        cv2.imwrite(f"{SAVE_DIR}/frame_{i:03d}.png", img)
        time.sleep(0.02)

    cam.shutDown()

if __name__ == "__main__":
    main()
