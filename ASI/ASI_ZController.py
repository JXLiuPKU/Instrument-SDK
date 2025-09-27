from sc_hardware.ASI.ms2k import MS2000
import re

class ASIZStage:
    """
    锁焦用 Z 轴控制适配（5 nm 步进；不改外部接口）：
      - zMoveTo(z_um) / zMoveRel(dz_um) / zPosition() / zPos()
      - zAtTarget(tol_um) / wait() / zZero() / close() / shutDown()
    实现要点：
      * 连接后自动将 UM Z 设为 200000（= 5 nm/计数），设置后立即查询复核；
      * 运动/读位采用 MOVE/WHERE（随 UM 缩放），不使用 MOVREL（其单位固定 0.1 µm）。
    """
    def __init__(self, port: str = "COM3", baud: int = 9600, report: bool = True):
        self.ms = MS2000(port, baud, report)
        self.ms.connect_to_serial()

        # --- 1) 查询/设置 UM 到 5 nm，并建立本地换算 ---
        self.um_per_count = 0.1            # 回退默认（µm/计数）
        self._um_residual = 0.0            # ΣΔ 残差（单位 µm）
        self._um_units_per_mm = self._query_um_units_per_mm() or 10000
        # 5 nm/计数（200000 units/mm）
        # 10 nm/计数（100000 units/mm）
        # 20 nm/计数（50000 units/mm）
        self._apply_um_units_per_mm(200000)

    # ================== UM 相关（units/mm） ==================
    def _query_um_units_per_mm(self):
        self.ms.send_command("UM Z?")
        resp = self.ms.read_response()      # 可能是 ':A Z=200000' 或 ':Z=200000.000000 A'
        m = re.search(r'Z\s*=\s*(\d+)', resp or "")
        return int(m.group(1)) if m else None

    def _apply_um_units_per_mm(self, n_units_per_mm: int, save: bool = True):
        """下发 UM 并复核；成功则更新 um_per_count 与残差。"""
        # 已是目标则直接刷新本地
        if self._um_units_per_mm == n_units_per_mm:
            self.um_per_count = 1000.0 / float(self._um_units_per_mm)
            self._um_residual = 0.0
            return

        # 下发设置
        self.ms.send_command(f"UM Z={int(n_units_per_mm)}")
        self.ms.read_response()
        if save:
            # 某些固件对“SS Z”不生效，使用“SS”保存全部
            self.ms.send_command("SS")
            self.ms.read_response()

        # 复核
        new_n = self._query_um_units_per_mm()
        if new_n and new_n == n_units_per_mm:
            self._um_units_per_mm = new_n
        # 刷新本地换算（无论成功与否都以实际 UM 为准）
        self.um_per_count = 1000.0 / float(self._um_units_per_mm)
        self._um_residual = 0.0

    # ================== 底层 WHERE/MOVE（UM 单位） ==================
    def _where_units(self) -> int:
        """返回 Z 轴当前位置（单位：UM 整数计数）。"""
        self.ms.send_command("W Z")
        resp = self.ms.read_response()    #
        m = re.search(r':A\s+(-?\d+)', resp or "")
        if not m:
            self.ms.send_command("WHERE Z")
            resp = self.ms.read_response()
            m = re.search(r':A\s+(-?\d+)', resp or "")
        return int(m.group(1)) if m else 0

    def _move_abs_units(self, target_units: int):
        """绝对移动到 UM 计数位（MOVE），并等待完成。"""
        self.ms.send_command(f"M Z={int(target_units)}")
        self.ms.read_response()
        self.ms.wait_for_device()

    # ================== 对外接口（保持原名/语义不变） ==================
    def zMoveRel(self, dz_um: float) -> None:
        """相对移动（单位 µm）。保留 ΣΔ：凑满 1 个 UM 计数才下发。"""
        acc = self._um_residual + float(dz_um)
        step = self.um_per_count              # 5 nm 时 step=0.005 µm
        # |acc| >= 1*step 才能形成整数计数
        counts = int(acc / step) if acc >= 0 else -int((-acc) / step)
        if counts != 0:
            cur_units = self._where_units()
            self._move_abs_units(cur_units + counts)
            acc -= counts * step
        self._um_residual = acc

    def zMoveTo(self, z_um: float) -> None:
        """绝对移动（单位 µm）。"""
        # 由 µm → UM整数计数 = round(z_um * UM / 1000)
        units = int(round(float(z_um) * self._um_units_per_mm / 1000.0))
        self._move_abs_units(units)
        self._um_residual = 0.0

    def zPosition(self) -> float:
        """当前 Z 位置（µm）。"""
        units = self._where_units()
        return units * (1000.0 / self._um_units_per_mm)

    def zPos(self) -> float:
        return self.zPosition()

    def zAtTarget(self, tol_um: float = 0.02) -> bool:
        """到位判断：控制器非忙即认为到位。"""
        return (not self.ms.is_axis_busy("Z"))

    def wait(self) -> None:
        self.ms.wait_for_device()

    def zZero(self) -> None:
        """当前位置设为 Z=0。"""
        self.ms.send_command("ZERO Z")
        self.ms.read_response()

    def close(self) -> None:
        self.ms.disconnect_from_serial()

    def shutDown(self) -> None:
        self.close()
