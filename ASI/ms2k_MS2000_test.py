# sc_hardware/ASI/ms2k_MS2000_test.py
from sc_hardware.ASI.ms2k import MS2000

PORT = "COM3"
BAUD = 9600
UM = 10  # 1 μm = 10 ASI units

def run():
    ms = MS2000(PORT, BAUD, report=True)
    ms.connect_to_serial()
    if not ms.is_open():
        print("串口未打开，退出"); return

    p0 = ms.get_position_um("Z")
    print(f"[Z] p0 = {p0:.3f} μm")

    ms.moverel_axis("Z", +50*UM);  ms.wait_for_device()
    p1 = ms.get_position_um("Z");  print(f"[Z] p1 = {p1:.3f} μm")

    ms.moverel_axis("Z", -50*UM);  ms.wait_for_device()
    p2 = ms.get_position_um("Z");  print(f"[Z] p2 = {p2:.3f} μm")

    ms.move_axis("Z", int((p0 + 100)*UM));  ms.wait_for_device()
    p3 = ms.get_position_um("Z");  print(f"[Z] p3 = {p3:.3f} μm  (target {p0+100:.3f} μm)")

    ms.send_command("ZERO Z"); ms.read_response(); ms.wait_for_device()
    pz = ms.get_position_um("Z"); print(f"[Z] after ZERO = {pz:.3f} μm")

    ms.disconnect_from_serial()

if __name__ == "__main__":
    run()
