import os, time
import numpy as np
import scipy
import scipy.optimize
from scipy.ndimage import label, center_of_mass
from ctypes import c_ubyte, addressof
from math import sqrt
import cv2
import gxipy as gx

def fitAFunctionLS(data, params, fn):
    errorfn = lambda p: np.ravel(fn(*p)(*np.indices(data.shape)) - data)
    [result, cov_x, infodict, mesg, success] = scipy.optimize.leastsq(
        errorfn, params, full_output=1, maxfev=1000)
    return [result, (1 <= success <= 4)]

def fixedEllipticalGaussian(background, height, center_x, center_y, width_x, width_y):
    return lambda x, y: background + height * np.exp(
        -(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) * 2)

def fitFixedEllipticalGaussian(data, params_0=None):
    if params_0 is None:
        params = [np.min(data), np.max(data), 0.5 * data.shape[0], 0.5 * data.shape[1], 6, 6]
    else:
        params = params_0
    return fitAFunctionLS(data, params, fixedEllipticalGaussian)

# ====== Daheng 原生相机封装 ======
class _GX:
    def __init__(self, camera_index=1, exposure_us=14000.0, gain=0.0):
        self.dm = gx.DeviceManager()
        dev_num, _ = self.dm.update_all_device_list()
        if dev_num == 0:
            raise RuntimeError("No Daheng camera found.")

        self.cam = self.dm.open_device_by_index(int(camera_index))
        if self.cam is None:
            raise RuntimeError("Failed to open camera (maybe already opened by another app).")

        fc = self.cam.get_remote_device_feature_control()
        fc.get_enum_feature("PixelFormat").set("Mono8")
        fc.get_float_feature("ExposureTime").set(float(exposure_us))
        fc.get_float_feature("Gain").set(float(gain))

        self.cvt = self.dm.create_image_format_convert()
        self.cvt.set_dest_format(gx.GxPixelFormatEntry.MONO8)

        self.streaming = False

    def setAOI(self, x_start, y_start, width, height):

        fc = self.cam.get_remote_device_feature_control()

        fc.get_int_feature("Width").set(width)
        fc.get_int_feature("Height").set(height)
        fc.get_int_feature("OffsetX").set(x_start)
        fc.get_int_feature("OffsetY").set(y_start)

    def stream_on(self):
        self.cam.stream_on()
        self.streaming = True

    def stream_off(self):
        if self.streaming:
            self.cam.stream_off()
            self.streaming = False

    def captureImage(self):
        raw = self.cam.data_stream[0].get_image()
        buf_size = self.cvt.get_buffer_size_for_conversion(raw)
        out = (c_ubyte * buf_size)()
        self.cvt.convert(raw, addressof(out), buf_size, False)
        arr = np.frombuffer(out, dtype=np.uint8, count=buf_size)
        return arr.reshape(raw.frame_data.height, raw.frame_data.width)

    def shutDown(self):
        try:
            self.cam.stream_off()
        except:
            pass
        try:
            self.cam.close_device()
            # gx.gx_close_lib()
        except:
            pass

class CameraQPD:
    """
    与 uc480Camera.CameraQPD 完全一致的接口/语义：
      - qpdScan(reps) -> [power, x_offset, y_offset]
      - getImage() -> (image, x1, y1, x2, y2, sigma)
      - adjustAOI(dx,dy) / changeFitMode(mode) / adjustZeroDist(inc)
      - setAOI() / getZeroDist() / shutDown()
    """
    def __init__(self, camera_id=0, fit_mutex=False, x_width=176, y_width=176,
                 sigma=3, offset_file=False, background=0):
        self.offset_file = offset_file          # ROI
        self.background = background
        self.fit_mode = 1
        self.fit_mutex = fit_mutex
        self.fit_size = int(2 * sigma)
        self.image = None
        self.sigma = sigma
        self.x_off1 = 0.0; self.y_off1 = 0.0
        self.x_off2 = 0.0; self.y_off2 = 0.0

        self.min_sep = 60                      # 两光斑最小间距（px）
        self.win_r = 10                        # 拟合/质心细化窗口半径（px）
        self.ema_beta = 0.65                   # 时间平滑系数 (0~1, 越小越稳)
        self.max_jump = 0.1                    # 单帧最大允许跳变（px）
        self._prev_xy = None                   # 上一帧平滑后的 (x1,y1,x2,y2)

        # 打开 Daheng 相机
        self.cam = _GX(camera_index=camera_id)

        # 读/设 AOI 起点
        with open(self.offset_file) as fp:
            self.x_start, self.y_start = map(int, fp.readline().split(",")[:2])

        self.x_width = x_width
        self.y_width = y_width
        self.setAOI()

    def setAOI(self):
        self.cam.setAOI(self.x_start, self.y_start, self.x_width, self.y_width)
        try:
            self.cam.stream_on()
        except:
            pass

    def adjustAOI(self, dx, dy):
        self.x_start += int(dx); self.y_start += int(dy)
        if self.x_start < 0: self.x_start = 0
        if self.y_start < 0: self.y_start = 0
        self.setAOI()

    def adjustZeroDist(self, inc):
        self.zero_dist += float(inc)

    def getZeroDist(self):
        return self.zero_dist

    def changeFitMode(self, mode):
        self.fit_mode = int(mode)

    def shutDown(self):
        if self.offset_file:
            with open(self.offset_file, "w") as fp:
                fp.write(f"{int(self.x_start)},{int(self.y_start)}")
        self.cam.shutDown()

    # === 抓图 ===
    def capture(self):
        try:
            self.image = self.cam.captureImage()
        except:
            self.cam.stream_on()
            self.image = self.cam.captureImage()
        return self.image

    def getImage(self):
        return [self.image, self.x_off1, self.y_off1, self.x_off2, self.y_off2, self.sigma]

    def _two_centroids(self, img):
        # 轻度平滑减少单像素噪声
        blur = cv2.GaussianBlur(img, (0, 0), 0.8)

        # 自适应阈值（mean+3σ）
        t = int(np.ceil(blur.mean() + 3.0 * blur.std()))
        t = max(10, min(t, 250))
        _, mask = cv2.threshold(blur, t, 255, cv2.THRESH_TOZERO)
        bw = (mask > 0)

        lab, n = label(bw)
        if n < 2:
            return None

        H, W = img.shape
        yy, xx = np.indices((H, W))
        comps = []
        for i in range(1, n + 1):
            m = (lab == i)
            if m.sum() < 10:
                continue
            w = img[m].astype(float)
            s = w.sum()
            cx = (xx[m] * w).sum() / s
            cy = (yy[m] * w).sum() / s
            comps.append((s, cx, cy))  # 用总亮度排序更鲁棒

        if len(comps) < 2:
            return None

        comps.sort(reverse=True)
        (s1, x1, y1), (s2, x2, y2) = comps[:2]

        # 最小分离约束
        if (x1 - x2) ** 2 + (y1 - y2) ** 2 < self.min_sep ** 2:
            return None

        # 从左到右排序，稳定顺序
        return sorted([(x1, y1), (x2, y2)], key=lambda p: p[0])

    def _fit_peak_elliptic(self, img, x, y):
        r = int(self.win_r)
        x0, y0 = int(round(x)), int(round(y))
        xs = slice(max(0, x0 - r), min(img.shape[1], x0 + r + 1))
        ys = slice(max(0, y0 - r), min(img.shape[0], y0 + r + 1))
        sub = img[ys, xs].astype(float)
        if sub.size < 9 or sub.max() < 25:
            return x, y

        # 初值：质心在局部窗坐标
        yy, xx = np.indices(sub.shape)
        w = sub
        sx = (xx * w).sum() / w.sum()
        sy = (yy * w).sum() / w.sum()

        # 也可用二阶矩估宽度；这里用 self.sigma 做简化
        sigma0 = float(self.sigma)
        p0 = [sub.min(), sub.max(), sy, sx, sigma0, sigma0]

        try:
            [p, ok] = fitFixedEllipticalGaussian(sub, p0)
            if not ok: raise RuntimeError
            cx = xs.start + p[3]  # 注意我们保持 (x,y) 返回
            cy = ys.start + p[2]
            return float(cx), float(cy)
        except Exception:
            # 拟合失败：回退质心
            return xs.start + sx, ys.start + sy

    def singleQpdScan(self):
        data = self.capture().copy()
        power = float(data.max())
        if power < 25:
            return [0.0, 0.0, 0.0]

        # 阈值到正：与原逻辑一致
        mean_cv2, std_cv2 = cv2.meanStdDev(data)
        thr = int(np.ceil(mean_cv2 + 3 * std_cv2))
        _, img1 = cv2.threshold(data, thr, 255, cv2.THRESH_TOZERO)

        # 1) 预定位（连通域质心）
        pts = self._two_centroids(img1)  # [(x1,y1),(x2,y2)]
        if pts is None:
            # 没找到两个可靠团：沿用上一帧（或给 0）
            if self._prev_xy is None:
                return [power, 0.0, 0.0]
            x1, y1, x2, y2 = self._prev_xy
        else:
            (x1, y1), (x2, y2) = pts

            # 2) 椭圆高斯局部拟合（失败回退质心）
            x1, y1 = self._fit_peak_elliptic(img1, x1, y1)
            x2, y2 = self._fit_peak_elliptic(img1, x2, y2)

            # 3) 单帧异常拒绝（>max_jump 直接沿用上一帧）
            if self._prev_xy is not None:
                px1, py1, px2, py2 = self._prev_xy
                if sqrt((x1 - px1) ** 2 + (y1 - py1) ** 2) > self.max_jump:
                    x1, y1 = px1, py1
                if sqrt((x2 - px2) ** 2 + (y2 - py2) ** 2) > self.max_jump:
                    x2, y2 = px2, py2

            # 4) 时间 EMA 去抖
            if self._prev_xy is None:
                self._prev_xy = (x1, y1, x2, y2)
            else:
                a = self.ema_beta
                px1, py1, px2, py2 = self._prev_xy
                x1 = (1 - a) * px1 + a * x1
                y1 = (1 - a) * py1 + a * y1
                x2 = (1 - a) * px2 + a * x2
                y2 = (1 - a) * py2 + a * y2
                self._prev_xy = (x1, y1, x2, y2)

        # 更新给 UI
        self.x_off1, self.y_off1 = x1, y1
        self.x_off2, self.y_off2 = x2, y2

        # 保持你当前的“欧氏距离平方”输出
        x_offset = (x1 - x2) ** 2
        y_offset = (y1 - y2) ** 2
        return [power, float(round(x_offset, 4)), float(round(y_offset, 4))]

    def qpdScan(self, reps=3):
        power_total = 0.0; x_total = 0.0; y_total = 0.0
        reps = max(1, int(reps))
        for _ in range(reps):

            power, xo, yo = self.singleQpdScan()
            power_total += power; x_total += xo; y_total += yo

        return [power_total/reps, x_total/reps, y_total/reps]

