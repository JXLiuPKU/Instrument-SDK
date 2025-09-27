# dm_control_api.py
import bmc
import numpy as np
import torch
import os
from typing import Iterable, Sequence, Optional
from zernikepy import zernike_polynomials

from utils_FPWSC.gram_schmidt_transform import gram_schmidt_transform_, balance_rms_
from utils_FPWSC.hilbert_transform import hilbert_transform_
from utils_FPWSC.hilbert_transform import probe_hilbert_generation

class DMError(RuntimeError):
    pass

def DM_grid_command_to_double_vector(dm: bmc.BmcDm, DM_grid_command) -> bmc.DoubleVector:
    """
    MutilDM
    输入：12×12的二维数据（list/np.ndarray/torch.tensor）
    动作：去掉四个角 → 按行优先展平 → 剪裁到[0,1] → 返回 bmc.DoubleVector
    说明：仅用于“你手上确实是方阵网格”的场景；普通 140 一维帧不要用它。
    """
    w = dm.num_actuators_width()   # 12

    # 统一转换为numpy格式，依次进行tensor\list\numpy的判定
    if isinstance(DM_grid_command, torch.Tensor):
        DM_grid_command = DM_grid_command.detach().cpu().numpy()
    elif isinstance(DM_grid_command, np.ndarray):
        DM_grid_command = DM_grid_command
    else:
        DM_grid_command = np.asarray(DM_grid_command, dtype=float)

    DM_grid_command[0, 0] = np.nan
    DM_grid_command[0, -1] = np.nan
    DM_grid_command[-1, 0] = np.nan
    DM_grid_command[-1, -1] = np.nan

    # 行优先展平并过滤掉四角
    DM_flat_command = DM_grid_command.ravel()
    DM_flat_command = DM_flat_command[~np.isnan(DM_flat_command)]

    N = dm.num_actuators()

    if DM_flat_command.size != N:
        raise DMError(f"输入的DM电压数量={DM_flat_command.size}，但设备需要 {N} 个电压")

    DM_flat_command = np.clip(DM_flat_command, 0.0, 1.0)

    # 转 DoubleVector
    DM_flat_command_double_vector = bmc.DoubleVector()
    # DM_flat_command_double_vector.assign(DM_flat_command.tolist())
    DM_flat_command_double_vector.assign(DM_flat_command.size, 0.0)
    for i, v in enumerate(DM_flat_command):
        DM_flat_command_double_vector[i] = float(v)

    return DM_flat_command_double_vector

# 假如输入数据本就是一维的tensor、numpy、list，用此函数转为double_vector格式 ( 适用于加载flat_map )
def DM_flat_command_to_double_vector(dm: bmc.BmcDm, DM_flat_command) -> bmc.DoubleVector:
    """
    接收一维的 140 个命令（tensor/ndarray/list 均可），自动转为 float、拉平成 1D，并裁剪到 [0,1]
    返回 bmc.DoubleVector, 长度必须等于 dm.num_actuators()
    """

    N = dm.num_actuators()

    # 统一转换为numpy格式，依次进行tensor\list\numpy的判定
    if isinstance(DM_flat_command, torch.Tensor):
        DM_flat_command = DM_flat_command.detach().cpu().numpy()
    elif isinstance(DM_flat_command, np.ndarray):
        DM_flat_command = DM_flat_command
    else:
        DM_flat_command = np.asarray(DM_flat_command, dtype=float)

    if DM_flat_command.ndim != 1:
        DM_flat_command = np.asarray(DM_flat_command, dtype=float).reshape(-1)   # 拉平为 1D
        raise DMError(f"需要一维长度的数据帧，收到其余维度的数据，已拉平，但建议检查是否有错误")
    if DM_flat_command.size != N:
        raise DMError(f"需要一维长度 {N} 的数据帧，收到长度 {DM_flat_command.size}")

    DM_flat_command = np.clip(DM_flat_command, 0.0, 1.0)

    # 转 DoubleVector
    DM_flat_command_double_vector = bmc.DoubleVector()
    # DM_flat_command_double_vector.assign(DM_flat_command.tolist())
    DM_flat_command_double_vector.assign(DM_flat_command.size, 0.0)
    for i, v in enumerate(DM_flat_command):
        DM_flat_command_double_vector[i] = float(v)

    return DM_flat_command_double_vector

# 随机 DM 指令,基于zernike多项式随机组合
def generate_random_DM_grid_command() -> np.ndarray:
    """
    生成一个 12×12 的随机 DM 网格（numpy 格式），
    由 0~20 共 21 个 Zernike 模式随机线性组合而成。
    返回：np.ndarray，shape=(12,12)，dtype=float32（值域未限定，后续 send_DM_command 会裁剪到[0,1]）
    """
    #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 随机生成每个 Zernike 系数
    non_zero = torch.rand(21, device=device) < 0.9
    Rn = 12 * torch.rand(21, device=device)
    RA = torch.empty(21, device=device).uniform_(0.1, 0.8)
    RC = RA * (2 * Rn - 1) * non_zero
    RC[[0, 1, 2, 4]] = 0  # 去掉 piston/tiltXY/defocus

    RC = RC.unsqueeze(1).to(device)  # [21,1]

    Zernike_modes = torch.from_numpy(
        zernike_polynomials(mode=20, size=12, select='all', show=False)
    ).to(torch.float32).to(device)  # [21,12,12]

    DM_grid = torch.einsum('ijk,kl->ijl', Zernike_modes, RC).squeeze()   # [12,12]

    # 转 numpy，并归一化到 [0.2, 0.8]
    DM_grid_command_numpy = DM_grid.detach().cpu().numpy().astype(np.float32)
    vmin, vmax = float(DM_grid_command_numpy.min()), float(DM_grid_command_numpy.max())

    # 先归一到 [0,1]，再映射到 [0.2,0.8]
    DM_grid_command_numpy = (DM_grid_command_numpy - vmin) / (vmax - vmin)
    DM_grid_command_numpy = 0.2 + 0.6 * DM_grid_command_numpy

    # 裁剪到 [0.2, 0.8]，再谨慎一点，不然DM真炸了
    np.clip(DM_grid_command_numpy, 0.2, 0.8, out=DM_grid_command_numpy)

    return DM_grid_command_numpy

def generate_sinc_DM_probe_command(device, config_PAO, DFT_PAO_vars):
    """
    返回: (surf_0, surf_1, surf_2)  ->  每个都是 numpy.ndarray，shape=(12,12)，dtype=float32
    公式:
      fx = sinc(t/T), fy = sinc(t/T),  T=0.1,  t ∈ [-1,1], N=12
      surf_0 = fx ⊗ fy
      surf_1 = fx ⊗ Hilbert(fy)
      surf_2 = Hilbert(fx) ⊗ fy
    """
    N = 12
    T = 0.1

    t  = torch.linspace(-1.0, 1.0, config_PAO['pixel_params']['N_3'], dtype=torch.float32)
    fx = torch.sinc(t / T)
    fy = torch.sinc(t / T)

    # 施密特正交化
    fx, Hx = gram_schmidt_transform_(fx, hilbert_transform_(fx))
    fy, Hy = gram_schmidt_transform_(fy, hilbert_transform_(fy))

    sinc_probe_DM_command_0 = torch.outer(fx, fy)
    sinc_probe_DM_command_1 = torch.outer(fx, Hy)
    sinc_probe_DM_command_2 = torch.outer(Hx, fy)
    sinc_probe_DM_command_3 = torch.outer(Hx, Hy)

    # 统一 RMS，避免某一张能量太弱/太强
    sinc_probe_DM_command_0, sinc_probe_DM_command_1, sinc_probe_DM_command_2, sinc_probe_DM_command_3 = balance_rms_(
        sinc_probe_DM_command_0, sinc_probe_DM_command_1, sinc_probe_DM_command_2, sinc_probe_DM_command_3)

    # # 外积
    # sinc_probe_DM_command_0 = torch.outer(fx, fy).unsqueeze(0).unsqueeze(0).to(device)
    # sinc_probe_DM_command_1 = torch.outer(fx, hilbert_transform_(fy)).unsqueeze(0).unsqueeze(0).to(device)
    # sinc_probe_DM_command_2 = torch.outer(hilbert_transform_(fx), fy).unsqueeze(0).unsqueeze(0).to(device)
    # sinc_probe_DM_command_3 = torch.outer(hilbert_transform_(fx), fy).unsqueeze(0).unsqueeze(0).to(device)

    sinc_probe_0_DM_command_base = 100 * probe_hilbert_generation(DFT_PAO_vars, sinc_probe_DM_command_0.unsqueeze(0).unsqueeze(0).to(device)).squeeze()
    sinc_probe_1_DM_command_base = 100 * probe_hilbert_generation(DFT_PAO_vars, sinc_probe_DM_command_1.unsqueeze(0).unsqueeze(0).to(device)).squeeze()
    sinc_probe_2_DM_command_base = 100 * probe_hilbert_generation(DFT_PAO_vars, sinc_probe_DM_command_2.unsqueeze(0).unsqueeze(0).to(device)).squeeze()
    sinc_probe_3_DM_command_base = 100 * probe_hilbert_generation(DFT_PAO_vars, sinc_probe_DM_command_3.unsqueeze(0).unsqueeze(0).to(device)).squeeze()

    return sinc_probe_0_DM_command_base, sinc_probe_1_DM_command_base, sinc_probe_2_DM_command_base, sinc_probe_3_DM_command_base

def generate_flat_map_DM_command():
    """
        从 txt 读取 140 个值，填入 12×12 网格（去掉四个角）：
          - 前 10 个 → 第 0 行的 [1..10]
          - 中间 120 个 → 第 1..10 行的 [0..11]（行优先）
          - 最后 10 个 → 第 11 行的 [1..10]
        返回: torch.float32, shape = (1, 1, 12, 12)
        """
    FLAT_MAP_PATH = r"D:\Desktop\Thorlabs DM140\17DW015#042_FLAT_MAP_COMMANDS.txt"
    # 读取 txt
    DM_flat_map_command_list_values = []
    with open(FLAT_MAP_PATH, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            DM_flat_map_command_list_values.append(float(s))

    if len(DM_flat_map_command_list_values) != 140:
        raise ValueError(f"文件中应有 140 个元素，实际为 {len(DM_flat_map_command_list_values)}")

    DM_flat_map_grid_command = torch.zeros((12, 12), dtype=torch.float32)

    k = 0
    # 顶行：跳过两角 → 填 10 个
    DM_flat_map_grid_command[0, 1:12 - 1] = torch.tensor(DM_flat_map_command_list_values[k:k + 10], dtype=torch.float32)
    k += 10

    # 中间 10 行（1..10）：每行 12 个，共 120 个
    mid_count = 10 * 12  # 120
    mid_vals = torch.tensor(DM_flat_map_command_list_values[k:k + mid_count], dtype=torch.float32)
    DM_flat_map_grid_command[1:12 - 1, :] = mid_vals.reshape(10, 12)
    k += mid_count

    # 底行：跳过两角 → 填 10 个
    DM_flat_map_grid_command[12 - 1, 1:12 - 1] = torch.tensor(DM_flat_map_command_list_values[k:k + 10], dtype=torch.float32)

    # 输出形状 [12, 12]
    return DM_flat_map_grid_command

# 打开 DM
def open_DM(serial_number: str, maps_path: Optional[str] = None, log_path: Optional[str] = None) -> bmc.BmcDm:

    """
    如果使用zernike_command，必须加载calibration文件，最好填充log_path,其余情况只需要serial_number
    示例:
    dm = open_DM(SERIAL,log_path=LOG_PATH)
    err_code = dm.load_calibration_file('C:\Program Files\Boston Micromachines\Calibration\Sample_Multi_OLC1_CAL.mat')
    if err_code:
        print('Zernike test failed.')
        raise Exception(dm.error_string(err_code))
    """

    dm = bmc.BmcDm()

    if log_path:
        abs_path = os.path.abspath(log_path)
        err_code = dm.configure_log(abs_path, bmc.BMC_LOG_DEBUG)
        if err_code:
            raise DMError(f"configure_log: {dm.error_string(err_code)}")

    if maps_path:
        err_code = dm.set_maps_path(maps_path)
        if err_code:
            raise DMError(f"set_maps_path: {dm.error_string(err_code)}")

    err_code = dm.open_dm(serial_number)
    if err_code:
        raise DMError(f"open_dm: {dm.error_string(err_code)}")

    return dm

def send_zero_DM_command(dm: bmc.BmcDm)-> None:
    w = dm.num_actuators_width()
    DM_zero_command = np.zeros((w, w), dtype=float)
    DM_zero_command_double_vector = DM_grid_command_to_double_vector(dm, DM_zero_command)
    err_code = dm.send_data(DM_zero_command_double_vector)
    if err_code: raise DMError(dm.error_string(err_code))

# 发送DM指令
def send_DM_command(dm: bmc.BmcDm, DM_command, custom_mapping: Optional[Sequence[int]] = None) -> None:
    """
    输入一般是一维的 140 个命令（DoubleVector）。
    若输入为二维 12×12(list/ndarray/tensor)，转 DM_grid_command_to_double_vector 并得到 double vector
    若输入为一维（list/ndarray/tensor），转 DM_flat_command_to_double_vector 并得到 double vector
    最终一律以 DoubleVector 下发。
    """

    # 统一转为DoubleVector, 执行完判定模块最终输出DM_flat_command_double_vector
    if isinstance(DM_command, bmc.DoubleVector):
        DM_flat_command_double_vector = DM_command
        # print(f"[send_dm_data] 收到 DoubleVector，len={len(DM_flat_command_double_vector)}")
    else:
        # 先强制转为numpy,tensor使用detach、cpu、numpy来执行转换，list使用asarray来转换
        # 执行转换是因为后续需要通过维度判定来决定走哪条路，但是list没有维度属性，所以最好还是先全都直接转为numpy
        if isinstance(DM_command, torch.Tensor):
            DM_command = DM_command.detach().cpu().numpy()
        else:
            DM_command = np.asarray(DM_command)
    # -----------------------------------------------------------------------
        if DM_command.ndim == 2 and DM_command.shape == (dm.num_actuators_width(), dm.num_actuators_width()):
            print("[send_dm_data] 输入为 2D 网格 -> 调用 DM_grid_command_to_double_vector")
            DM_flat_command_double_vector = DM_grid_command_to_double_vector(dm, DM_command)
        elif DM_command.ndim == 1:
            print("[send_dm_data] 输入为 1D 序列 -> 调用 DM_flat_command_to_double_vector")
            DM_flat_command_double_vector = DM_flat_command_to_double_vector(dm, DM_command)
        else:
            raise DMError(f"不支持的 data 形状 {DM_command.shape}；"
                          f"请传一维长度 {dm.num_actuators()} 或 {dm.num_actuators_width()}×{dm.num_actuators_width()} 方阵")

    # 传递command, 只判断映射是否默认,默认则调用dm.send_data直接传递命令，不默认则调用dm.send_data_custom_mapping
    if custom_mapping is None:
        err_code = dm.send_data(DM_flat_command_double_vector)
        if err_code: raise DMError(dm.error_string(err_code))
    else:
        iv = bmc.IntVector(); iv.assign(list(map(int, custom_mapping)))
        err_code = dm.send_data_custom_mapping(DM_flat_command_double_vector, iv)
        if err_code: raise DMError(dm.error_string(err_code))

# 按 Zernike 系数加载面形
def send_DM_zernike_command(dm: bmc.BmcDm, zernike_coeffs: Sequence[float], dm_diameter: int = 0,
    set_options: int = 0) -> None:

    """
    - Zernike 系数采用 OSA 索引。
    - 需要 DoubleVector，内部已从 list 转换；set_surface 直接使用返回的 surface。
    - dm_diameter设置为0.则使用默认驱动器区域:dm.num_actuators_width()-3,使用全孔径请设置dm_diameter = dm.num_actuators_width()-1
    - options = 0, 无作用；
    """

    w = dm.num_actuators_width()
    dm_diameter = dm.num_actuators_width() - 1  # 这里使用的是全孔径，如果使用推荐孔径，就是默认的 dm_diameter: int = 0
    # 转double vector
    zernike_coeffs_double_vector = bmc.DoubleVector()
    # zernike_coeffs_double_vector.assign([float(x) for x in zernike_coeffs])  # 显式转成 double vector 并填入
    zernike_coeffs_double_vector.assign(len(zernike_coeffs), 0.0)
    for i, v in enumerate(zernike_coeffs):
        zernike_coeffs_double_vector[i] = float(v)

    err_code, surface = dm.zernike_surface(zernike_coeffs_double_vector, dm_diameter, 0)
    # print(type(surface))
    #
    # surface_np = np.array(surface, dtype=np.float64)
    #
    # print(type(surface_np), surface_np.shape, surface_np.dtype)
    # print(surface_np[:10])  # 打印前 10 个数看看

    if err_code:
        raise DMError(f"zernike_surface: {dm.error_string(err_code)}")

    err_code = dm.set_surface(surface, w, w)
    if err_code:
        raise DMError(f"set_surface: {dm.error_string(err_code)}")

# 加载 flat_map
def send_DM_flat_map_command(dm: bmc.BmcDm, txt_path: str) -> None:
    """
    从 txt 读取“flat map”命令并下发到 DM。
    - txt: 每行一个浮点数（0~1），总数应等于 dm.num_actuators()（你的 12×12−4=140）
    - 若发现越界值会先裁剪到 [0,1]
    - 依赖默认映射顺序（默认映射由设备 profile 决定）
    """

    # 读取txt文件中DM的电压
    flat_map_list_values: list[float] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            flat_map_list_values.append(float(s))

    # 检查读取是否有误
    N = dm.num_actuators()
    if len(flat_map_list_values) != N:
        raise DMError(f"flat_map 数量应为 {N}，但文件里是 {len(flat_map_list_values)}")

    # 转double vector
    flat_map_command_double_vector = bmc.DoubleVector()
    # flat_map_command_double_vector.assign([float(x) for x in flat_map_list_values])  # 显式转成 double 并填入
    flat_map_command_double_vector.assign(len(flat_map_list_values), 0.0)
    for i, v in enumerate(flat_map_list_values):
        flat_map_command_double_vector[i] = float(v)

    err_code = dm.send_data(flat_map_command_double_vector)
    if err_code:
        raise DMError(f"send_data(flat_map): {dm.error_string(err_code)}")

# 关闭 DM
def close_DM(dm: bmc.BmcDm) -> None:
    err_code = dm.close_dm()
    if err_code:
        raise DMError(f"close_dm: {dm.error_string(err_code)}")
