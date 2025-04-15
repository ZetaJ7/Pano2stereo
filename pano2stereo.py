import torch
import numpy as np
import threading
from threading import Thread
import queue
import time
import math, os
import cv2
import logging
import subprocess
from pathlib import Path
from tqdm import tqdm
from depth_estimate import depth_estimation
from submodule.DepthAnythingv2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

from numba import cuda
import cupy as cp

# def image_coords_to_uv(image_width, image_height):
#     # 生成网格坐标 (x, y)
#     x = cp.arange(image_width)
#     y = cp.arange(image_height)
#     xx, yy = cp.meshgrid(x, y)
    
#     # 归一化到 [-1, 1]
#     u = (xx / (image_width - 1)) * 2 - 1
#     v = (yy / (image_height - 1)) * -2 + 1
    
#     return cp.stack([u, v], axis=-1)  # 返回形状 (H, W, 2)

model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}, 
    }

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

@cuda.jit(device=True)
def _cartesian_to_sphere(x,y,z):
    r = math.sqrt(x*x + y*y + z*z)
    theta = math.acos(z / r) if r != 0 else 0.0
    phi = math.atan2(y, x) if r != 0 else 0.0
    return r, theta, phi
    
@cuda.jit
def image_coords_to_sphere_kernel(d_u, d_v, d_output):
    i, j = cuda.grid(2)
    if i < d_u.shape[0] and j < d_u.shape[1]:
        lon = d_u[i,j] * math.pi
        lat = d_v[i,j] * (math.pi/2)
        # 笛卡尔坐标计算...
        x = cp.cos(lat) * cp.cos(lon)
        y = cp.cos(lat) * cp.sin(lon)
        z = cp.sin(lat)
        r, theta, phi = _cartesian_to_sphere(x, y, z)
        # 将结果存储在输出数组中
        d_output[i,j,0] = r
        d_output[i,j,1] = theta
        d_output[i,j,2] = phi 

class StereoPano():
    def __init__(self, height, width, IPD, pro_radius, pixel, critical_depth, vit_size='vits', dataset='hypersim'):
        logging.info('Initializing StereoPano Generator')
        t1 = cv2.getTickCount()
        self.image = None
        self.IPD = IPD
        self.width = width
        self.height = height
        self.pro_radius = pro_radius
        self.pixel = pixel
        self.critical_depth = critical_depth
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model = DepthAnythingV2(**{**model_configs[vit_size], 'max_depth': 20})
        self.model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{vit_size}.pth', map_location=self.device))
        self.model = self.model.to(self.device).eval()
        self.sphere_coords = self.image_coords_to_sphere(width, height)
        # Generator model activate
        random = np.array(np.random.rand(1, 3, height, width), dtype=np.float32)
        self.model.infer_image(random) # First time inference will be slow
        t2 = cv2.getTickCount()
        logging.info('[DepthAnythingV2] Model loaded: {}-{}'.format(vit_size,dataset))
        logging.info(f'[DepthAnythingV2] Initialization time: {(t2 - t1) / cv2.getTickFrequency()} seconds')
        
    def image_coords_to_sphere(self, image_width, image_height, x=None, y=None):
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
            x = cp.arange(image_width)
            y = cp.arange(image_height)
            xx, yy = cp.meshgrid(x, y)
        else:
            xx, yy = cp.array(x), cp.array(y)
        
        # 归一化到 [-1, 1]
        u = (xx / (image_width - 1)) * 2 - 1  # 经度方向
        v = (yy / (image_height - 1)) * -2 + 1  # 纬度方向
        
        # 计算经度纬度
        lon = u * cp.pi           # 经度范围: -π ~ π
        lat = v * (cp.pi / 2)     # 纬度范围: -π/2 ~ π/2
        
        # 计算笛卡尔坐标 (x, y, z)
        x_xyz = cp.cos(lat) * cp.cos(lon)
        y_xyz = cp.cos(lat) * cp.sin(lon)
        z_xyz = cp.sin(lat)
        
        # 计算球面坐标 (r, θ, φ)
        r = cp.sqrt(x_xyz**2 + y_xyz**2 + z_xyz**2)
        safe_r = cp.where(r == 0, 1e-12, r)  # 避免除以零
        theta = cp.arccos(z_xyz / safe_r)    # 极角 [0, π]
        phi = cp.arctan2(y_xyz, x_xyz)       # 方位角 [-π, π]
        
        # 处理原点情况
        theta = cp.where(r == 0, 0.0, theta)
        phi = cp.where(r == 0, 0.0, phi)
        
        # 合并结果
        sphere_coords = cp.stack([r, theta, phi], axis=-1)
        t2 = cv2.getTickCount()
        time = (t2 - t1) / cv2.getTickFrequency()
        logging.info(f"Image to sphere coordinates time: {time:.4f} seconds")
        return sphere_coords.squeeze()  # 移除多余维度

    def sphere_to_image_coords(self, sphere_coords, image_width, image_height):
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
        sphere_coords = cp.asarray(sphere_coords)
        r = sphere_coords[..., 0]
        theta = sphere_coords[..., 1]  # 极角 [0, π]
        phi = sphere_coords[..., 2]    # 方位角 [-π, π]
        
        # 规范化角度范围
        theta = cp.clip(theta, 0, cp.pi)            # 极角限制在 [0, π]
        phi = (phi + cp.pi) % (2 * cp.pi) - cp.pi    # 方位角限制在 [-π, π]
        
        # 计算纬度（lat）和经度（lon）
        lat = cp.pi/2 - theta  # 纬度范围 [-π/2, π/2]
        lon = phi              # 经度范围 [-π, π]
        
        # 归一化到 [-1, 1]
        u = lon / cp.pi        # 经度方向：-π~π → -1~1
        v = lat / (cp.pi/2)    # 纬度方向：-π/2~π/2 → -1~1
        
        # 转换为浮点图像坐标
        x_float = (u + 1) * 0.5 * (image_width - 1)
        y_float = (1 - v) * 0.5 * (image_height - 1)
        
        # 四舍五入并转换为整数
        x = cp.round(x_float).astype(cp.int32)
        y = cp.round(y_float).astype(cp.int32)
        
        # 合并坐标并确保在图像范围内
        image_coords = cp.stack([
            cp.clip(x, 0, image_width - 1),
            cp.clip(y, 0, image_height - 1)
        ], axis=-1)
        t2 = cv2.getTickCount()
        time = (t2 - t1) / cv2.getTickFrequency()
        logging.info(f"Sphere to image coordinates time: {time:.4f} seconds")
        return image_coords

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

    #     clinder_l = cp.stack([r_l, theta_l, h_l], axis=-1)
    #     clinder_r = cp.stack([r_r, theta_r, h_r], axis=-1)

    #     return clinder_l, clinder_r

    def normalize_channel(self, data, percentile=99):
        """将单通道浮点数据归一化到 [0, 1] 范围，并处理异常值"""
        # 过滤 NaN 和无穷大
        data = cp.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 基于百分位数截断极值（保留前99%的数据范围）
        max_val = cp.percentile(data, percentile)
        min_val = cp.percentile(data, 100 - percentile)
        data = cp.clip(data, min_val, max_val)
        
        # 线性归一化
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        return normalized

    def repair_black_regions(self, image):
        t1 = cv2.getTickCount()
        # 转换到BGR格式用于OpenCV处理
        image = cp.asnumpy(image).astype(np.uint8)
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # 创建掩膜（黑色区域为255，其他为0）
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mask = np.uint8(gray == 0) * 255
        # 使用Telea算法进行修复
        repaired = cv2.inpaint(img_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        t2 = cv2.getTickCount()
        time = (t2 - t1) / cv2.getTickFrequency()
        logging.debug(f"Repair black regions time: {time:.4f} seconds")
        return cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)
    
    # def gpu_inpaint(self, img_gpu, mask_gpu):
    #     # 获取无效点坐标
    #     invalid_points = cp.argwhere(mask_gpu)
        
    #     # 生成坐标网格
    #     y_grid, x_grid = cp.indices(mask_gpu.shape)
        
    #     # 计算距离矩阵 (H,W,N)
    #     distances = cp.sqrt(
    #         (y_grid[..., None] - invalid_points[:,0])**2 + 
    #         (x_grid[..., None] - invalid_points[:,1])**2
    #     )
        
    #     # 获取最近邻索引 (H,W)
    #     min_indices = cp.argmin(distances, axis=2)
        
    #     # 提取有效像素值 (N,3)
    #     valid_values = img_gpu[invalid_points[:,0], invalid_points[:,1], :]
        
    #     # 生成修复结果 (H,W,3)
    #     repaired = valid_values[min_indices]
        
    #     # 维度对齐的掩膜处理
    #     mask_3d = mask_gpu[..., None]  # 形状(H,W,1)
        
    #     # 广播规则验证
    #     return cp.where(mask_3d, repaired, img_gpu)

    def make_output_dir(self, base_dir='results/output'):
        index = 0
        while True:
            target_dir = f"{base_dir}_{index:03d}" if index >= 0 else base_dir
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
                break
            index += 1
        return target_dir

    def save_depth_map(self, depth, output_path):
        depth = depth.get()  # GPU to CPU
        min,max = depth.min(), depth.max()
        logging.info(f'Depth min: {min}, max: {max}')
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_normalized.astype(cp.uint8)
        cv2.imwrite('{}/depth.png'.format(output_path), depth_uint8)

    def sphere2pano_vectorized(self, sphere_coords, z_vals):
        # Runs on GPU with cupy
        # t1 = cv2.getTickCount()
        R = cp.asarray(sphere_coords[..., 0])
        theta = cp.asarray(sphere_coords[..., 1])
        phi = cp.asarray(sphere_coords[..., 2])

        mask = z_vals < self.critical_depth
        ratio = cp.clip(self.IPD / cp.where(mask, z_vals, cp.inf), -1.0, 1.0)
        delta = cp.arcsin(ratio)

        phi_l = (phi + delta) % (2 * cp.pi)
        phi_r = (phi - delta) % (2 * cp.pi)

        # t2 = cv2.getTickCount()
        # time = (t2 - t1) / cv2.getTickFrequency()
        # logging.info(f"Pano-fix calculate time: {time:.4f} seconds")
        return (
            cp.stack([R, theta, phi_l], axis=-1),
            cp.stack([R, theta, phi_r], axis=-1)
        )

    def generate_stereo_pair_vectorized(self, rgb_data, depth_data):
        # Speed up with multi-threading
        logging.info("=============== Generating stereo pair(Threading) ================")
        t1 = cv2.getTickCount()
        
        # 计算左右视差修正后的各像素的球坐标
        sphere_l, sphere_r = self.sphere2pano_vectorized(self.sphere_coords, depth_data)
        
        # 转换回图像坐标系
        coords_l = self.sphere_to_image_coords(sphere_l, self.width, self.height)
        coords_r = self.sphere_to_image_coords(sphere_r, self.width, self.height)
        xl, yl = coords_l[..., 0], coords_l[..., 1]
        xr, yr = coords_r[..., 0], coords_r[..., 1]

        # 初始化输出图像和深度缓冲区
        left = cp.zeros_like(rgb_data)
        right = cp.zeros_like(rgb_data)
        z_left = cp.full((self.height, self.width), cp.inf)
        z_right = cp.full((self.height, self.width), cp.inf)

        def process_eye_wrapper(x_dest, y_dest, result, z_buffer):
            """线程安全处理包装器"""
            src_y, src_x, dest_y, dest_x, z_vals = process_eye(x_dest, y_dest)
            with threading.Lock():
                result[dest_y, dest_x] = rgb_data[src_y, src_x]
                z_buffer[dest_y, dest_x] = z_vals
        
        def process_eye(x_dest, y_dest):
            """优化后的处理逻辑"""
            valid = (x_dest >= 0) & (x_dest < self.width) & (y_dest >= 0) & (y_dest < self.height)
            dest_indices = y_dest[valid] * self.width + x_dest[valid]
            source_idx = cp.where(valid)
            z_values = depth_data[source_idx]
            
            sorted_idx = cp.argsort(z_values)
            _, unique_idx = cp.unique(dest_indices[sorted_idx], return_index=True)
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

        #repair black regions
        # left_repaired = self.repair_black_regions(left)
        # right_repaired = self.repair_black_regions(right)
        # logging.debug('left,right type:{}'.format(type(left_repaired),type(right_repaired)))

        return left, right

    def generate_stereo_pair(self, rgb_array, depth):
        return self.generate_stereo_pair_vectorized(rgb_array, depth)
    
    def parallel_inpaint(self, left, right):
        from multiprocessing import Pool
        with Pool(2) as p:
            return p.starmap(self.repair_black_regions, [(left,), (right,)])

def main():
    # ============= 1. Load Image ================
    # img_path = "3D60/1_color_0_Left_Down_0.0.png"
    # img_path = "3D60/2_color_0_Left_Down_0.0.png"
    img_path = "Lab_data/Lab1_360P.png"
    # img_path = "Lab_data/Lab2.jpg"
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"无法读取图像文件: {img_path}")
    logging.info(f"Icput Image: {img_path}")
    bgr_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr_array is None:
        raise FileNotFoundError(f"无法读取图像文件: {img_path}")
    rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
    height, width = rgb_array.shape[:2]
    logging.info(f"Image Size: {width}x{height}")
    
    # Transfer to GPU memory
    d_rgb = cp.asarray(rgb_array, dtype=cp.float32)  # 转换为cupy数组
    
    IPD = 0.032                     # 视环半径(m) 
    PRO_RADIUS = 1                  # 视环投影半径(m) 
    PIXEL = 4448                    # 赤道图像宽度 4448(pixel)
    CRITICAL_DEPTH = 9999           # 临界深度(m)
    GENERATE_VIDEO = 0              # 是否生成视频

    CRITICAL_DEPTH_CAL = cal_critical_depth(IPD, PIXEL)
    CRITICAL_DEPTH = min(CRITICAL_DEPTH, CRITICAL_DEPTH_CAL)
    logging.info('[Stereo Parameters]\nCircular Radius: %sm\cprojection Radius: %sm\cpixel on Equator: %s\nCritical Depth: %sm\n===============================' % (IPD, PRO_RADIUS, PIXEL, CRITICAL_DEPTH))

    # ============= 2. Depth Estimation ================
    # Class initialization
    generator = StereoPano(height=height,width=width,IPD=IPD, pro_radius=PRO_RADIUS, pixel=PIXEL, critical_depth = CRITICAL_DEPTH,vit_size='vits',dataset='hypersim')

    t1 = cv2.getTickCount()
    depth = generator.model.infer_image(rgb_array) # HxW depth map in meters in cpu().numpy()
    depth = cp.asarray(depth, dtype=cp.float32)  # 转换为cupy数组
    t2 = cv2.getTickCount()
    logging.debug(f'[DepthAnythingV2] Inference time: {(t2 - t1) / cv2.getTickFrequency()} seconds')
    assert d_rgb.shape[:2] == depth.shape[:2]

    left, right = generator.generate_stereo_pair(d_rgb, depth)
    t3 = cv2.getTickCount()
    logging.debug(f'[StereoPano] Generate stereo pair time: {(t3 - t2) / cv2.getTickFrequency()} seconds')
    
    # Repair black regions
    left_repaired = generator.repair_black_regions(left)
    right_repaired = generator.repair_black_regions(right)
    t4 = cv2.getTickCount()

    logging.debug(f'[StereoPano] Repair black regions time: {(t4 - t3) / cv2.getTickFrequency()} seconds')
    
    logging.info(f'[StereoPano] Frame Processing time: {(t4 - t1) / cv2.getTickFrequency()} seconds')


    #================== 3. Save Results ================
    left = cp.asnumpy(left)  # 转换为numpy数组
    right = cp.asnumpy(right)  # 转换为numpy数组
    left_bgr = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
    right_bgr = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
    output_dir = generator.make_output_dir()
    generator.save_depth_map(depth, output_dir)
    cv2.imwrite("{}/left.png".format(output_dir), left_bgr)
    cv2.imwrite("{}/right.png".format(output_dir), right_bgr)

    left_repaired = cp.asnumpy(left_repaired)  # 转换为numpy数组
    right_repaired = cp.asnumpy(right_repaired)  # 转换为numpy数组
    cv2.imwrite("{}/left_repaired.png".format(output_dir), cv2.cvtColor(left_repaired, cv2.COLOR_RGB2BGR))
    cv2.imwrite("{}/right_repaired.png".format(output_dir), cv2.cvtColor(right_repaired, cv2.COLOR_RGB2BGR))

    output_dir = Path(output_dir).absolute()
    try:
        # Red-Cyan 3D anaglyph
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

        # Stereo image (left-right)
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
            # generate video
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




    

