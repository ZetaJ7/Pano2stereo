import numpy as np
import math, os
import pyexr
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

RADIUS = 0.32           # 视环半径 32cm
PRO_RADIUS = 1       # 视环投影半径 1m
CRITICAL_DEPTH = 25     # 临界深度 25m

# def image_coords_to_uv(image_width, image_height):
#     # 生成网格坐标 (x, y)
#     x = np.arange(image_width)
#     y = np.arange(image_height)
#     xx, yy = np.meshgrid(x, y)
    
#     # 归一化到 [-1, 1]
#     u = (xx / (image_width - 1)) * 2 - 1
#     v = (yy / (image_height - 1)) * -2 + 1
    
#     return np.stack([u, v], axis=-1)  # 返回形状 (H, W, 2)

def image_coords_to_sphere(image_width, image_height, x=None, y=None):
    """
    将图像坐标 (x, y) 转换为球面坐标 (r, θ, φ)
    
    参数:
        image_width (int)  : 图像宽度（像素数）
        image_height (int) : 图像高度（像素数）
        x (int/array)      : 像素x坐标（可选，默认生成全图网格）
        y (int/array)      : 像素y坐标（可选，默认生成全图网格）
        
    返回:
        sphere_coords (array): 球面坐标数组，形状 (H, W, 3) 或单个坐标 (3,)
    """
    # 生成全图网格或处理单个坐标
    if x is None or y is None:
        x = np.arange(image_width)
        y = np.arange(image_height)
        xx, yy = np.meshgrid(x, y)
    else:
        xx, yy = np.array(x), np.array(y)
    
    # 归一化到 [-1, 1]
    u = (xx / (image_width - 1)) * 2 - 1  # 经度方向
    v = (yy / (image_height - 1)) * -2 + 1  # 纬度方向
    
    # 计算经度纬度
    lon = u * np.pi           # 经度范围: -π ~ π
    lat = v * (np.pi / 2)     # 纬度范围: -π/2 ~ π/2
    
    # 计算笛卡尔坐标 (x, y, z)
    x_xyz = np.cos(lat) * np.cos(lon)
    y_xyz = np.cos(lat) * np.sin(lon)
    z_xyz = np.sin(lat)
    
    # 计算球面坐标 (r, θ, φ)
    r = np.sqrt(x_xyz**2 + y_xyz**2 + z_xyz**2)
    safe_r = np.where(r == 0, 1e-12, r)  # 避免除以零
    theta = np.arccos(z_xyz / safe_r)    # 极角 [0, π]
    phi = np.arctan2(y_xyz, x_xyz)       # 方位角 [-π, π]
    
    # 处理原点情况
    theta = np.where(r == 0, 0.0, theta)
    phi = np.where(r == 0, 0.0, phi)
    
    # 合并结果
    sphere_coords = np.stack([r, theta, phi], axis=-1)
    return sphere_coords.squeeze()  # 移除多余维度

def sphere_to_image_coords(sphere_coords, image_width, image_height):
    """
    将球坐标 (r, theta, phi) 转换为整型图像坐标 (x, y)
    假设使用等距柱状投影（Equirectangular Projection）
    
    参数:
        sphere_coords (array-like): 形状为 (..., 3)，包含 [r, theta, phi]
        image_width (int)         : 图像宽度（像素数）
        image_height (int)        : 图像高度（像素数）
        
    返回:
        image_coords (array): 图像坐标数组，形状为 (..., 2)，数据类型为int32
    """
    sphere_coords = np.asarray(sphere_coords)
    r = sphere_coords[..., 0]
    theta = sphere_coords[..., 1]  # 极角 [0, π]
    phi = sphere_coords[..., 2]    # 方位角 [-π, π]
    
    # 规范化角度范围
    theta = np.clip(theta, 0, np.pi)            # 极角限制在 [0, π]
    phi = (phi + np.pi) % (2 * np.pi) - np.pi    # 方位角限制在 [-π, π]
    
    # 计算纬度（lat）和经度（lon）
    lat = np.pi/2 - theta  # 纬度范围 [-π/2, π/2]
    lon = phi              # 经度范围 [-π, π]
    
    # 归一化到 [-1, 1]
    u = lon / np.pi        # 经度方向：-π~π → -1~1
    v = lat / (np.pi/2)    # 纬度方向：-π/2~π/2 → -1~1
    
    # 转换为浮点图像坐标
    x_float = (u + 1) * 0.5 * (image_width - 1)
    y_float = (1 - v) * 0.5 * (image_height - 1)
    
    # 四舍五入并转换为整数
    x = np.round(x_float).astype(np.int32)
    y = np.round(y_float).astype(np.int32)
    
    # 合并坐标并确保在图像范围内
    image_coords = np.stack([
        np.clip(x, 0, image_width - 1),
        np.clip(y, 0, image_height - 1)
    ], axis=-1)
    
    return image_coords

def sphere2pano(sphere, z, r=RADIUS):
    if z >= CRITICAL_DEPTH:
        return sphere, sphere

    R = sphere[..., 0]
    theta = sphere[..., 1]  # 极角 θ
    phi = sphere[..., 2]    # 方位角 φ

    # 使用 numpy 处理向量化计算
    ratio = r / z
    ratio = np.clip(ratio, -1.0, 1.0)  # 防止浮点溢出
    
    delta = np.arcsin(ratio)  # 计算偏移量
    
    # 修正 phi 计算公式 
    phi_l = (phi + delta) % (2 * np.pi)  # 确保在 [0, 2π] 范围内
    phi_r = (phi - delta) % (2 * np.pi)  # 确保在 [0, 2π] 范围内

    sphere_l = np.stack([R, theta, phi_l], axis=-1)
    sphere_r = np.stack([R, theta, phi_r], axis=-1)
    
    return sphere_l, sphere_r

# def clinder2pano(clinder, z, r=RADIUS):
#     """
#     柱坐标 (r, θ, h)
#     """
#     if z >= CRITICAL_DEPTH:
#         return clinder, clinder

#     r = clinder[..., 0]         # 半径 r
#     theta = clinder[..., 1]     # 极角 θ
#     h = clinder[..., 2]         # 高度 z
    
#     r_l = r
#     r_r = r

#     h_l = h
#     h_r = h

#     theta_l = (theta + math.arcsin(r/z)) % (2*math.pi)
#     theta_r = (theta - math.arcsin(r/z)) % (2*math.pi)

#     clinder_l = np.stack([r_l, theta_l, h_l], axis=-1)
#     clinder_r = np.stack([r_r, theta_r, h_r], axis=-1)

#     return clinder_l, clinder_r

def normalize_channel(data, percentile=99):
    """将单通道浮点数据归一化到 [0, 1] 范围，并处理异常值"""
    # 过滤 NaN 和无穷大
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 基于百分位数截断极值（保留前99%的数据范围）
    max_val = np.percentile(data, percentile)
    min_val = np.percentile(data, 100 - percentile)
    data = np.clip(data, min_val, max_val)
    
    # 线性归一化
    normalized = (data - min_val) / (max_val - min_val + 1e-8)
    return normalized

def generate_stereo_pair(rgb_data, depth_data):
    H, W = rgb_data.shape[:2]
    sphere_coords = image_coords_to_sphere(W, H)  # 生成球面坐标
    
    # 明确初始化类型为 uint8
    left = np.zeros_like(rgb_data)
    right = np.zeros_like(rgb_data)
    z_left = np.full((H, W), np.inf)
    z_right = np.full((H, W), np.inf)

    for h0 in tqdm(range(H)):
        for w0 in range(W):
            z = depth_data[h0, w0]
            if z <= 0 or z >= CRITICAL_DEPTH:
                continue

            sphere = sphere_coords[h0, w0]
            sphere_l, sphere_r = sphere2pano(sphere, z)

            coord_l = sphere_to_image_coords(sphere_l, W, H)
            xl, yl = coord_l[0], coord_l[1]
            coord_r = sphere_to_image_coords(sphere_r, W, H)
            xr, yr = coord_r[0], coord_r[1]

            if 0 <= xl < W and 0 <= yl < H:
                if z < z_left[yl, xl]:
                    left[yl, xl] = rgb_data[h0, w0]
                    z_left[yl, xl] = z

            if 0 <= xr < W and 0 <= yr < H:
                if z < z_right[yr, xr]:
                    right[yr, xr] = rgb_data[h0, w0]
                    z_right[yr, xr] = z

    return left, right

def bilinear_interpolate(image, x, y):
    h, w = image.shape[:2]
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, w-1)
    x1 = np.clip(x1, 0, w-1)
    y0 = np.clip(y0, 0, h-1)
    y1 = np.clip(y1, 0, h-1)

    Ia = image[y0, x0]
    Ib = image[y1, x0]
    Ic = image[y0, x1]
    Id = image[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa[..., None] * Ia + wb[..., None] * Ib + wc[..., None] * Ic + wd[..., None] * Id

if __name__ == "__main__":
    img_path = "3D60/1_color_0_Left_Down_0.0.png"
    depth_path = img_path.replace("color", "depth").replace(".png", ".exr")
    
    exr_data = pyexr.read(depth_path)
    depth = exr_data[..., 0].copy()

    bgr_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr_array is None:
        raise FileNotFoundError(f"无法读取图像文件: {img_path}")
    rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)

    assert rgb_array.shape[:2] == depth.shape[:2]

    left, right = generate_stereo_pair(rgb_array, depth)
    
    left_bgr = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
    right_bgr = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/left.png", left_bgr)
    cv2.imwrite("output/right.png", right_bgr)





    

