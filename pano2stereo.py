import numpy as np
import threading
import math, os
import pyexr
import matplotlib.pyplot as plt
import cv2
import logging
import subprocess
from pathlib import Path
from tqdm import tqdm
from depth_estimate import depth_estimation

import multiprocessing
from multiprocessing import Process,Queue
from multiprocessing.shared_memory import SharedMemory

# def image_coords_to_uv(image_width, image_height):
#     # 生成网格坐标 (x, y)
#     x = np.arange(image_width)
#     y = np.arange(image_height)
#     xx, yy = np.meshgrid(x, y)
    
#     # 归一化到 [-1, 1]
#     u = (xx / (image_width - 1)) * 2 - 1
#     v = (yy / (image_height - 1)) * -2 + 1
    
#     return np.stack([u, v], axis=-1)  # 返回形状 (H, W, 2)

def logging_setup():
    file_handler = logging.FileHandler('pano2stereo.log', mode='w')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
           file_handler,logging.StreamHandler()
        ]
    )
    logging.info("Logging setup complete.")

def cal_critical_depth(r, pixel):
    return r/(math.sin(2*math.pi/pixel))

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
    t1 = cv2.getTickCount()
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
    t2 = cv2.getTickCount()
    time = (t2 - t1) / cv2.getTickFrequency()
    logging.info(f"Image to sphere coordinates time: {time:.4f} seconds")
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
    t1 = cv2.getTickCount()
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
    t2 = cv2.getTickCount()
    time = (t2 - t1) / cv2.getTickFrequency()
    logging.info(f"Sphere to image coordinates time: {time:.4f} seconds")
    return image_coords

def sphere2pano(sphere, z, r, critical_depth):
    t1 = cv2.getTickCount()
    if z >= critical_depth:
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

    # fix theta
    # theta_l = math.atan((math.sqrt(z**2 - r**2) /z) * math.tan(theta)) 
    theta_l = theta
    theta_r = theta_l

    sphere_l = np.stack([R, theta_l, phi_l], axis=-1)
    sphere_r = np.stack([R, theta_r, phi_r], axis=-1)
    t2 = cv2.getTickCount()
    time = (t2 - t1) / cv2.getTickFrequency()
    logging.info(f"Sphere to pano time: {time:.4f} seconds")
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

def repair_black_regions(image):
    # 转换到BGR格式用于OpenCV处理
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # 创建掩膜（黑色区域为255，其他为0）
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = np.uint8(gray == 0) * 255
    # 使用Telea算法进行修复
    repaired = cv2.inpaint(img_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)

def make_output_dir(base_dir='results/output'):
    index = 0
    while True:
        target_dir = f"{base_dir}_{index:03d}" if index >= 0 else base_dir
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            break
        index += 1
    return target_dir

def save_depth_map(depth, output_path):
    min,max = depth.min(), depth.max()
    logging.info(f'Depth min: {min}, max: {max}')
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)
    cv2.imwrite('{}/depth.png'.format(output_path), depth_uint8)

def sphere2pano_vectorized(sphere_coords, z_vals, r, critical_depth):
    t1 = cv2.getTickCount()
    R = sphere_coords[..., 0]
    theta = sphere_coords[..., 1]
    phi = sphere_coords[..., 2]

    mask = z_vals < critical_depth
    ratio = np.clip(r / np.where(mask, z_vals, np.inf), -1.0, 1.0)
    delta = np.arcsin(ratio)

    phi_l = (phi + delta) % (2 * np.pi)
    phi_r = (phi - delta) % (2 * np.pi)

    t2 = cv2.getTickCount()
    time = (t2 - t1) / cv2.getTickFrequency()
    logging.info(f"Sphere to pano vectorized time: {time:.4f} seconds")
    return (
        np.stack([R, theta, phi_l], axis=-1),
        np.stack([R, theta, phi_r], axis=-1)
    )

def generate_stereo_pair_vectorized(rgb_data, depth_data,r, critical_depth):
    # Speed up with multi-threading
    logging.info("=============== Generating stereo pair(Threading) ================")
    t1 = cv2.getTickCount()
    H, W = rgb_data.shape[:2]
    sphere_coords = image_coords_to_sphere(W, H)
    
    # 计算左右投影坐标
    sphere_l, sphere_r = sphere2pano_vectorized(sphere_coords, depth_data, r, critical_depth)
    
    # 转换到图像坐标系
    coords_l = sphere_to_image_coords(sphere_l, W, H)
    coords_r = sphere_to_image_coords(sphere_r, W, H)
    xl, yl = coords_l[..., 0], coords_l[..., 1]
    xr, yr = coords_r[..., 0], coords_r[..., 1]

    # 初始化输出图像和深度缓冲区
    left = np.zeros_like(rgb_data)
    right = np.zeros_like(rgb_data)
    z_left = np.full((H, W), np.inf)
    z_right = np.full((H, W), np.inf)

    def process_eye_wrapper(x_dest, y_dest, result, z_buffer):
        """线程安全处理包装器"""
        src_y, src_x, dest_y, dest_x, z_vals = process_eye(x_dest, y_dest)
        with threading.Lock():
            result[dest_y, dest_x] = rgb_data[src_y, src_x]
            z_buffer[dest_y, dest_x] = z_vals
    
    def process_eye(x_dest, y_dest):
        """优化后的处理逻辑"""
        valid = (x_dest >= 0) & (x_dest < W) & (y_dest >= 0) & (y_dest < H)
        dest_indices = y_dest[valid] * W + x_dest[valid]
        source_idx = np.where(valid)
        z_values = depth_data[source_idx]
        
        sorted_idx = np.argsort(z_values)
        _, unique_idx = np.unique(dest_indices[sorted_idx], return_index=True)
        selected = sorted_idx[unique_idx]
        
        return (source_idx[0][selected], 
                source_idx[1][selected],
                y_dest[valid][selected],
                x_dest[valid][selected],
                z_values[selected])

    # 创建并启动线程
    threads = []
    threads.append(threading.Thread(target=process_eye_wrapper, 
                                 args=(xl, yl, left, z_left)))
    threads.append(threading.Thread(target=process_eye_wrapper,
                                 args=(xr, yr, right, z_right)))
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    t2 = cv2.getTickCount()
    time = (t2 - t1) / cv2.getTickFrequency()
    logging.info(f"Panoramic processing TOTAL time: {time:.4f} seconds")
    return left, right

def generate_stereo_pair_multiprocess(rgb_data, depth_data, r, critical_depth):
    # Speed up with multiprocessing
    logging.info("=============== Generating stereo pair (Multi-process) ===================")
    t1 = cv2.getTickCount()
    H, W = rgb_data.shape[:2]
    
    # 生成球面坐标（主进程计算）
    sphere_coords = image_coords_to_sphere(W, H)
    sphere_l, sphere_r = sphere2pano_vectorized(sphere_coords, depth_data, r, critical_depth)
    
    # 转换为图像坐标
    xl, yl = sphere_to_image_coords(sphere_l, W, H)[..., 0], sphere_to_image_coords(sphere_l, W, H)[..., 1]
    xr, yr = sphere_to_image_coords(sphere_r, W, H)[..., 0], sphere_to_image_coords(sphere_r, W, H)[..., 1]

    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / cv2.getTickFrequency()
    logging.info(f"Coordinates translate TOTAL time: {time1:.4f} seconds")
    # 创建共享内存
    shm_rgb = SharedMemory(create=True, size=rgb_data.nbytes)
    shm_depth = SharedMemory(create=True, size=depth_data.nbytes)
    shm_left = SharedMemory(create=True, size=rgb_data.nbytes)
    shm_right = SharedMemory(create=True, size=rgb_data.nbytes)
    t3 = cv2.getTickCount()
    time2 = (t3 - t2) / cv2.getTickFrequency()
    logging.info(f"Shared memory allocation time: {time2:.4f} seconds")


    # 将数据复制到共享内存
    np.copyto(np.ndarray(rgb_data.shape, dtype=rgb_data.dtype, buffer=shm_rgb.buf), rgb_data)
    np.copyto(np.ndarray(depth_data.shape, dtype=depth_data.dtype, buffer=shm_depth.buf), depth_data)
    t4 = cv2.getTickCount()
    time3 = (t4 - t3) / cv2.getTickFrequency()
    logging.info(f"Data copy to shared memory time: {time3:.4f} seconds")

    # 创建进程间通信队列
    result_queue = Queue()

    # 启动左眼处理进程
    p_left = Process(target=process_eye_worker,
                    args=('left', shm_rgb.name, shm_depth.name, 
                         shm_left.name, xl, yl, W, H, result_queue))
    # 启动右眼处理进程
    p_right = Process(target=process_eye_worker,
                     args=('right', shm_rgb.name, shm_depth.name,
                          shm_right.name, xr, yr, W, H, result_queue))
    t5 = cv2.getTickCount()
    time4 = (t5 - t4) / cv2.getTickFrequency()
    logging.info(f"Process creation time: {time4:.4f} seconds")
    
    p_left.start()
    p_right.start()
    
    # 等待子进程完成
    p_left.join()
    p_right.join()
    
    # 重建结果数组
    left = np.ndarray((H, W, 3), dtype=rgb_data.dtype, buffer=shm_left.buf).copy()
    right = np.ndarray((H, W, 3), dtype=rgb_data.dtype, buffer=shm_right.buf).copy()
    t6 = cv2.getTickCount()
    time5 = (t6 - t5) / cv2.getTickFrequency()
    logging.info(f"StereoPairs Generation time: {time5:.4f} seconds")

    # 释放共享内存
    shm_rgb.close()
    shm_depth.close()
    shm_left.close()
    shm_right.close()
    shm_rgb.unlink()
    shm_depth.unlink()
    shm_left.unlink()
    shm_right.unlink()

    t7 = cv2.getTickCount()
    time6 = (t7 - t6) / cv2.getTickFrequency()
    logging.info(f"Memory clear time: {time6:.4f} seconds")
    return left, right

def process_eye_worker(side, shm_rgb_name, shm_depth_name, 
                      shm_output_name, x_dest, y_dest, W, H, queue):
    """工作进程处理单侧投影"""
    try:
        # 连接到共享内存
        shm_rgb = SharedMemory(name=shm_rgb_name)
        shm_depth = SharedMemory(name=shm_depth_name)
        shm_output = SharedMemory(name=shm_output_name)
        
        # 重建numpy数组视图
        rgb = np.ndarray((H, W, 3), dtype=np.uint8, buffer=shm_rgb.buf)
        depth = np.ndarray((H, W), dtype=np.float32, buffer=shm_depth.buf)
        output = np.ndarray((H, W, 3), dtype=np.uint8, buffer=shm_output.buf)
        z_buffer = np.full((H, W), np.inf)
        
        # 处理逻辑
        valid = (x_dest >= 0) & (x_dest < W) & (y_dest >= 0) & (y_dest < H)
        dest_indices = y_dest[valid] * W + x_dest[valid]
        source_idx = np.where(valid)
        z_values = depth[source_idx]
        
        sorted_idx = np.argsort(z_values)
        _, unique_idx = np.unique(dest_indices[sorted_idx], return_index=True)
        selected = sorted_idx[unique_idx]
        
        src_y = source_idx[0][selected]
        src_x = source_idx[1][selected]
        dest_y = y_dest[valid][selected]
        dest_x = x_dest[valid][selected]
        
        output[dest_y, dest_x] = rgb[src_y, src_x]
        queue.put(f"{side} process completed")
        
    finally:
        # 关闭共享内存连接
        shm_rgb.close()
        shm_depth.close()
        shm_output.close()


def generate_stereo_pair(rgb_array, depth, r, critical_depth, method=0):
    if method == 0:     # Multi-threading
        return generate_stereo_pair_vectorized(rgb_array, depth, r, critical_depth)
    elif method == 1:   # Multi-processing
        return generate_stereo_pair_multiprocess(rgb_array, depth, r, critical_depth)

def main():
    # img_path = "3D60/1_color_0_Left_Down_0.0.png"
    # img_path = "3D60/2_color_0_Left_Down_0.0.png"
    img_path = "Lab_data/Lab1_360P.png"
    # img_path = "Lab_data/Lab2.jpg"
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"无法读取图像文件: {img_path}")
    logging.info(f"Input Image: {img_path}")
    
    RADIUS = 0.032                  # 视环半径(m) 
    PRO_RADIUS = 1                  # 视环投影半径(m) 
    PIXEL = 4448                    # 赤道图像宽度 4448(pixel)
    CRITICAL_DEPTH = 9999           # 临界深度(m)
    GENERATE_VIDEO = 0              # 是否生成视频

    CRITICAL_DEPTH_CAL = cal_critical_depth(RADIUS, PIXEL)
    CRITICAL_DEPTH = min(CRITICAL_DEPTH, CRITICAL_DEPTH_CAL)
    logging.info('[Stereo Parameters]\nCircular Radius: %sm\nProjection Radius: %sm\nPixel on Equator: %s\nCritical Depth: %sm\n===============================' % (RADIUS, PRO_RADIUS, PIXEL, CRITICAL_DEPTH))

    # Depth From GT
    # depth_path = img_path.replace("color", "depth").replace(".png", ".exr")
    # exr_data = pyexr.read(depth_path)
    # depth = exr_data[..., 0].copy()

    # Depth From Estimation
    # max_depth: 20 for indoor model, 80 for outdoor model
    # encoder: 'vits', 'vitb', 'vitl' for small/base/large model
    # dataset: 'hypersim' for indoor model, 'vkitti' for outdoor model
    depth = depth_estimation(cv2.imread(img_path), max_depth=20, encoder='vitb', dataset='hypersim')  


    bgr_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr_array is None:
        raise FileNotFoundError(f"无法读取图像文件: {img_path}")
    rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)

    assert rgb_array.shape[:2] == depth.shape[:2]
    height, width = rgb_array.shape[:2]
    logging.info(f"Image Size: {width}x{height}")

    left,right = generate_stereo_pair(rgb_array, depth, r=RADIUS, critical_depth=CRITICAL_DEPTH,method=0)
    
    left_bgr = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
    right_bgr = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)

    output_dir = make_output_dir()
    save_depth_map(depth, output_dir)
    cv2.imwrite("{}/left.png".format(output_dir), left_bgr)
    cv2.imwrite("{}/right.png".format(output_dir), right_bgr)

    # Repair black regions
    left_repaired = repair_black_regions(left)
    right_repaired = repair_black_regions(right)
    cv2.imwrite("{}/left_repaired.png".format(output_dir), cv2.cvtColor(left_repaired, cv2.COLOR_RGB2BGR))
    cv2.imwrite("{}/right_repaired.png".format(output_dir), cv2.cvtColor(right_repaired, cv2.COLOR_RGB2BGR))

    # 获取输出目录绝对路径
    output_dir = Path(output_dir).absolute()

  
    try:
        # 执行立体图像生成
        subprocess.run([
            "StereoscoPy",
            "-S", "5", "0",
            "-a",
            "-m", "color",
            "--cs", "red-cyan",
            "--lc", "rgb",
            str(output_dir/"left_repaired.png"),
            str(output_dir/"right_repaired.png"),
            str(output_dir/"red_cyan.jpg")
        ], check=True)

        # 执行水平拼接
        subprocess.run([
            "ffmpeg",
            "-y",  # 覆盖已有文件
            "-i", str(output_dir/"left_repaired.png"),
            "-i", str(output_dir/"right_repaired.png"),
            "-filter_complex",
            "[0:v]scale={}:{}[img1];[img1][1:v]hstack".format(
                int(width), int(height)),
            str(output_dir/"stereo.jpg")
        ], check=True)

        if GENERATE_VIDEO:      
            # 生成循环视频
            logging.info("Generating video...")
            subprocess.run([
                "ffmpeg",
                "-y",
                "-loop", "1",
                "-i", str(output_dir/"stereo.jpg"),
                "-c:v", "libx264",
                "-t", "60",
                "-pix_fmt", "yuv420p",
                "-vf", "fps=30",
                str(output_dir/"stereo_output.mp4")
            ], check=True)

    except subprocess.CalledProcessError as e:
        logging.info(f"Failure: {e.returncode}\n{e.stderr}")
    except Exception as e:
        logging.info(f"Error: {str(e)}")
    else:
        logging.info("Panoramic to Stereo conversion completed successfully.")
        
    logging.info(f"Output Dir: {output_dir}")

if __name__ == "__main__":
    logging_setup()
    main()




    

