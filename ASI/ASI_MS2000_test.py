# asi_zcontroller_quick_test.py
from sc_hardware.ASI.ASI_ZController import ASIZStage

# 固定测试参数（按需改端口/波特率；不想改就用 ASIZStage 的默认）
PORT = "COM3"
BAUD = 9600

STEP_REL = 50.0     # 相对位移（μm）
STEP_ABS = 100.0    # 绝对位移增量（μm）

z = ASIZStage(port=PORT, baud=BAUD, report=True)

# 读初始位置
p0 = z.zPosition()
print(f"[Z] p0 = {p0:.3f} μm")

# 相对 +STEP_REL
z.zMoveRel(+STEP_REL)
z.wait()
p1 = z.zPosition()
print(f"[Z] p1 = {p1:.3f} μm  (rel +{STEP_REL})  atTarget={z.zAtTarget()}")

# 相对 -STEP_REL
z.zMoveRel(-STEP_REL)
z.wait()
p2 = z.zPosition()
print(f"[Z] p2 = {p2:.3f} μm  (rel -{STEP_REL})  atTarget={z.zAtTarget()}")

# 绝对到 p0 + STEP_ABS
z.zMoveTo(p0 + STEP_ABS)
z.wait()
p3 = z.zPosition()
print(f"[Z] p3 = {p3:.3f} μm  (abs → {p0+STEP_ABS:.3f})  atTarget={z.zAtTarget()}")

# 回到初始位置 p0
z.zMoveTo(p0)
z.wait()
p4 = z.zPosition()
print(f"[Z] p4 = {p4:.3f} μm  (abs → {p0:.3f})  atTarget={z.zAtTarget()}")

# 归零
z.zZero()
z.wait()
pz = z.zPosition()
print(f"[Z] pos_zero = {pz:.3f} μm  (after ZERO)  atTarget={z.zAtTarget()}")

# 结束
z.close()
print("[Z] DONE.")
