# -*- coding: utf-8 -*-
import time
import serial

class MPBLaser:
    """
    MPB VFL 控制
      1) 打开激光器：open() 连接串口
      2) 使用功率控制：set_mode_apc()
      3) 功率设定：set_power_mw(mw)
      4) 开启激光：enable()
      5) 关闭激光：disable() + close()

    参考：DOC-04384R9
      - 7.1 命令格式（第18页）：命令以 CR 结束，成功/失败提示符 D >/F >
      - POWERENABLE（第22页）
      - SETPOWER/GETPOWER/POWER（第21–23页）
      - SETLDENABLE/GETLDENABLE（第20–21页）
    """

    def __init__(self, port: str, baudrate: int, timeout: float = 0.5):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None

    # 打开激光器,连接串口
    def open(self):
        if self.ser and self.ser.is_open:
            return
        self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)

    def close(self):
        if self.ser:
            self.ser.close()
            self.ser = None

    # 内部通信：发送一条命令，返回完整响应
    def send_command(self, cmd: str) -> str:
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("Serial not open. Call open().")
        self.ser.reset_input_buffer()
        self.ser.write((cmd + "\r").encode("ascii"))  # 串行同行指令需要以 CR(回车) 结尾, 也就是"\r"字符串
        time.sleep(0.05)
        return self.ser.read_all().decode("ascii", errors="ignore")

    def send_ok(self, resp: str) -> bool:
        return resp.strip().endswith("D >")  #

    # 使用功率控制（APC）, 我们基本上只会用功率控制
    def set_mode_apc(self):
        resp = self.send_command("POWERENABLE 1")
        if not self.send_ok(resp):
            raise RuntimeError(f"POWERENABLE 1 failed: {resp}")

    # 功率设定（mW）
    def set_power_mw(self, mw: float):
        resp = self.send_command(f"SETPOWER 0 {mw}")
        if not self.send_ok(resp):
            raise RuntimeError(f"SETPOWER failed: {resp}")

    # 开启激光（发射）
    def enable(self):
        resp = self.send_command("SETLDENABLE 1")
        if not self.send_ok(resp):
            raise RuntimeError(f"SETLDENABLE 1 failed: {resp}")

    # 关闭激光（发射关闭）
    def disable(self):
        resp = self.send_command("SETLDENABLE 0")
        if not self.send_ok(resp):
            raise RuntimeError(f"SETLDENABLE 0 failed: {resp}")

def MPBlaser_on(port="COM4", baudrate=9600, power_mw=200.0, wait_s=10, timeout=0.5):
    """
    打开激光器并开始输出光（APC模式 + 设定功率 + 使能发射）。

    Args:
        port: 串口名，"COM4"
        baudrate: 波特率，9600
        power_mw: 目标功率（mW），最低设置为200mw
        wait_s: 使能后等待的秒数（硬件有约3秒启动延时，这里最好多等一点）
        timeout: 串口读超时

    Returns:
        已连接且处于发射状态的 MPBLaser 对象（没有自动关闭）。
    """
    laser = MPBLaser(port=port, baudrate=baudrate, timeout=timeout)

    # 连接串口
    laser.open()
    print(f"[OK] Connected to {port} @ {baudrate} baud")

    # 切到功率控制
    laser.set_mode_apc()
    print("[OK] Mode -> APC")

    # 设定目标功率
    laser.set_power_mw(power_mw)
    print(f"[OK] Set power to {power_mw} mW")

    # 开启发射
    laser.enable()
    print("[OK] Laser ENABLED (emission on)")

    time.sleep(wait_s)

    # 没有关闭，所以需要将句柄输出，在外面继续控制它关闭
    return laser
