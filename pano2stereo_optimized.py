import torch
import numpy as np
import threading
import time
import math
import os
import cv2
import logging
import subprocess
from pathlib import Path
from tqdm import tqdm
from depth_estimate import depth_estimation
from submodule.DepthAnythingv2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from numba import cuda
import cupy as cp
from contextlib import contextmanager
import gc

# 尝试从配置文件加载参数，如果没有则使用默认值
try:
    import config
    PARAMS = {
        "IPD": config.IPD,
        "PRO_RADIUS": config.PRO_RADIUS,
        "PIXEL": config.PIXEL,
        "CRITICAL_DEPTH": config.CRITICAL_DEPTH,
        "GENERATE_VIDEO": config.GENERATE_VIDEO,
        "IMG_PATH": config.IMG_PATH,
        "VIT_SIZE": config.VIT_SIZE,
        "DATASET": config.DATASET,
        "OUTPUT_BASE": config.OUTPUT_BASE,
        "ENABLE_REPAIR": config.ENABLE_REPAIR,
        "ENABLE_POST_PROCESS": config.ENABLE_POST_PROCESS,
        "ENABLE_RED_CYAN": config.ENABLE_RED_CYAN,
        "TARGET_HEIGHT": config.TARGET_HEIGHT,
        "TARGET_WIDTH": config.TARGET_WIDTH
    }
    logging.info("Configuration loaded from config.py")
except ImportError:
    logging.warning("config.py not found, using default parameters")
    # 默认参数（如果没有配置文件）
    PARAMS = {
        "IPD": 0.032,                      # 视环半径(m)
        "PRO_RADIUS": 1,                   # 视环投影半径(m)
        "PIXEL": 4448,                     # 赤道图像宽度(pixel)
        "CRITICAL_DEPTH": 9999,            # 临界深度(m)
        "GENERATE_VIDEO": 0,               # 是否生成视频
        "IMG_PATH": "Data/Lab1_2k.png",       # 输入图片路径
        "VIT_SIZE": "vits",                # 深度模型类型
        "DATASET": "hypersim",             # 深度模型数据集
        "OUTPUT_BASE": "results/output",   # 输出目录基础名
        "ENABLE_REPAIR": True,             # 是否启用图像修复
        "ENABLE_POST_PROCESS": True,       # 是否启用后处理
        "ENABLE_RED_CYAN": False,          # 是否生成红青3D图（默认关闭以提升速度）
        "TARGET_HEIGHT": 540,              # 目标高度（降低分辨率提升速度）
        "TARGET_WIDTH": 960                # 目标宽度
    }

# ================== 性能监控 ==================
class PerformanceProfiler:
    def __init__(self):
        self.timings = {}
        self.active_timers = {}
    
    def reset(self):
        """重置计时器数据"""
        self.timings.clear()
        self.active_timers.clear()
    
    def get_timing(self, name):
        """获取指定计时器的最新时间"""
        if name in self.timings and self.timings[name]:
            return self.timings[name][-1]  # 返回最新的时间
        return None
    
    @contextmanager
    def timer(self, name):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(elapsed)
            
            # 立即输出并刷新日志
            message = f"[TIMER] {name}: {elapsed:.4f}s"
            logging.info(message)
            # 强制刷新所有handlers
            for handler in logging.getLogger().handlers:
                handler.flush()
    
    def get_summary(self):
        summary = "\n" + "="*50 + "\n[TIMER] Performance Summary:\n" + "="*50
        total_time = 0
        for name, times in self.timings.items():
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            total_time += sum(times)
            summary += f"\n{name}:"
            summary += f"\n  Average: {avg_time:.4f}s"
            summary += f"\n  Min: {min_time:.4f}s"
            summary += f"\n  Max: {max_time:.4f}s"
            summary += f"\n  Total: {sum(times):.4f}s"
            summary += f"\n  Count: {len(times)}"
        summary += f"\n{'-'*50}"
        summary += f"\nTotal execution time: {total_time:.4f}s"
        summary += f"\n{'='*50}"
        return summary

# 全局性能分析器
profiler = PerformanceProfiler()

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

def logging_setup():
    # 设置OpenCV日志级别，减少警告
    try:
        cv2.setLogLevel(3)  # 3=ERROR级别，减少警告信息
    except:
        pass  # 如果设置失败就跳过
    
    # 确保logs目录存在
    os.makedirs('logs', exist_ok=True)
    
    # 配置日志 - 强制刷新缓冲区
    file_handler = logging.FileHandler('logs/pano2stereo.log', mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 获取根logger并配置
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()  # 清除现有handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # 强制刷新
    logging.info("=== Pano2Stereo Optimized Processing Started ===")
    for handler in root_logger.handlers:
        handler.flush()

def cal_critical_depth(r, pixel):
    return r / (math.sin(2 * math.pi / pixel))

class OptimizedStereoPano():
    def __init__(self, height, width, IPD, pro_radius, pixel, critical_depth, vit_size='vits', dataset='hypersim'):
        logging.info('Initializing Optimized StereoPano Generator')
        self.IPD = IPD
        self.width = width
        self.height = height
        self.pro_radius = pro_radius
        self.pixel = pixel
        self.critical_depth = critical_depth
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # 立即加载模型（而不是延迟加载）
        logging.info("Loading depth estimation model...")
        self.model = DepthAnythingV2(**{**model_configs[vit_size], 'max_depth': 20})
        self.model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{vit_size}.pth', map_location=self.device))
        self.model = self.model.to(self.device).eval()
        logging.info(f'[DepthAnythingV2] Model loaded: {vit_size}-{dataset}')
        
        # 优化：预计算所有需要的坐标变换
        logging.info("Precomputing sphere coordinates...")
        self.sphere_coords = self._precompute_sphere_coords(width, height)
        # 预计算网格坐标用于快速采样
        self.grid_y, self.grid_x = cp.meshgrid(cp.arange(height), cp.arange(width), indexing='ij')
        
        # 模型预热（在初始化时完成）
        logging.info("Warming up model for optimal performance...")
        # 使用实际输入尺寸进行预热
        dummy_input = np.random.rand(height, width, 3).astype(np.float32)
        dummy_input = (dummy_input * 255).astype(np.uint8)  # 模拟真实图像格式
        _ = self.model.infer_image(dummy_input)
        # 清理预热时的内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Model warmup completed")
        
        logging.info('[OptimizedStereoPano] Initialization completed')

    def _precompute_sphere_coords(self, image_width, image_height):
        """优化的球面坐标预计算"""
        # 使用更高效的坐标生成
        x = cp.linspace(0, image_width - 1, image_width, dtype=cp.float32)
        y = cp.linspace(0, image_height - 1, image_height, dtype=cp.float32)
        xx, yy = cp.meshgrid(x, y)
        
        # 向量化计算
        u = (xx / (image_width - 1)) * 2 - 1
        v = (yy / (image_height - 1)) * -2 + 1
        lon = u * cp.pi
        lat = v * (cp.pi / 2)
        
        # 球面坐标计算
        x_xyz = cp.cos(lat) * cp.cos(lon)
        y_xyz = cp.cos(lat) * cp.sin(lon)
        z_xyz = cp.sin(lat)
        
        # 使用更稳定的计算方式
        r = cp.ones_like(x_xyz)  # 单位球面，r=1
        theta = cp.arccos(cp.clip(z_xyz, -1, 1))
        phi = cp.arctan2(y_xyz, x_xyz)
        
        return cp.stack([r, theta, phi], axis=-1)

    def sphere_to_image_coords(self, sphere_coords, image_width, image_height):
        """原版本的球面到图像坐标转换（更高效）"""
        with profiler.timer("Sphere_to_image_coords"):
            sphere_coords = cp.asarray(sphere_coords)
            r = sphere_coords[..., 0]
            theta = sphere_coords[..., 1]
            phi = sphere_coords[..., 2]
            theta = cp.clip(theta, 0, cp.pi)
            phi = (phi + cp.pi) % (2 * cp.pi) - cp.pi
            lat = cp.pi / 2 - theta
            lon = phi
            u = lon / cp.pi
            v = lat / (cp.pi / 2)
            x_float = (u + 1) * 0.5 * (image_width - 1)
            y_float = (1 - v) * 0.5 * (image_height - 1)
            x = cp.round(x_float).astype(cp.int32)
            y = cp.round(y_float).astype(cp.int32)
            image_coords = cp.stack([
                cp.clip(x, 0, image_width - 1),
                cp.clip(y, 0, image_height - 1)
            ], axis=-1)
            return image_coords

    def sphere2pano_vectorized(self, sphere_coords, z_vals):
        """原版本的高效球面视差计算"""
        with profiler.timer("Sphere2pano_vectorized"):
            R = cp.asarray(sphere_coords[..., 0])
            theta = cp.asarray(sphere_coords[..., 1])
            phi = cp.asarray(sphere_coords[..., 2])
            mask = z_vals < self.critical_depth
            ratio = cp.clip(self.IPD / cp.where(mask, z_vals, cp.inf), -1.0, 1.0)
            delta = cp.arcsin(ratio)
            phi_l = (phi + delta) % (2 * cp.pi)
            phi_r = (phi - delta) % (2 * cp.pi)
            return (
                cp.stack([R, theta, phi_l], axis=-1),
                cp.stack([R, theta, phi_r], axis=-1)
            )

    def generate_stereo_pair_vectorized(self, rgb_data, depth_data):
        """原版本的高效立体对生成"""
        with profiler.timer("Generate_stereo_pair_total"):
            logging.info("=============== Generating stereo pair (Original Method) ================")
            
            with profiler.timer("Sphere2pano_calculation"):
                sphere_l, sphere_r = self.sphere2pano_vectorized(self.sphere_coords, depth_data)
            
            with profiler.timer("Coordinate_mapping"):
                coords_l = self.sphere_to_image_coords(sphere_l, self.width, self.height)
                coords_r = self.sphere_to_image_coords(sphere_r, self.width, self.height)
            
            with profiler.timer("Image_sampling"):
                xl, yl = coords_l[..., 0], coords_l[..., 1]
                xr, yr = coords_r[..., 0], coords_r[..., 1]
                left = cp.zeros_like(rgb_data)
                right = cp.zeros_like(rgb_data)
                left[cp.arange(self.height)[:, None], cp.arange(self.width), :] = rgb_data[yl, xl, :]
                right[cp.arange(self.height)[:, None], cp.arange(self.width), :] = rgb_data[yr, xr, :]
            
            return left, right

    def generate_stereo_pair(self, rgb_array, depth):
        """统一的立体对生成入口"""
        return self.generate_stereo_pair_vectorized(rgb_array, depth)

    def repair_black_regions_fast(self, image):
        """快速黑色区域修复（可选）"""
        if not PARAMS["ENABLE_REPAIR"]:
            return cp.asnumpy(image)
            
        # 转换到CPU进行修复
        image_cpu = cp.asnumpy(image).astype(np.uint8)
        
        # 快速检测是否需要修复
        gray = cv2.cvtColor(image_cpu, cv2.COLOR_RGB2GRAY)
        black_pixels = np.sum(gray == 0)
        
        if black_pixels < gray.size * 0.01:  # 如果黑色像素少于1%，跳过修复
            return image_cpu
        
        # 使用更快的修复算法
        img_bgr = cv2.cvtColor(image_cpu, cv2.COLOR_RGB2BGR)
        mask = np.uint8(gray == 0) * 255
        repaired = cv2.inpaint(img_bgr, mask, inpaintRadius=2, flags=cv2.INPAINT_NS)  # 使用更快的算法
        return cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)

    def make_output_dir(self, base_dir=None):
        if base_dir is None:
            base_dir = PARAMS["OUTPUT_BASE"]
        index = 0
        while True:
            target_dir = f"{base_dir}_{index:03d}" if index >= 0 else base_dir
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
                break
            index += 1
        return target_dir

# =============== 优化的功能函数 ===============

def load_and_resize_image(img_path):
    """加载并可选地调整图像大小以提升速度"""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"File Not Found: {img_path}")
    
    bgr_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr_array is None:
        raise FileNotFoundError(f"Cannot Read file: {img_path}")
    
    rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
    
    # 可选：调整图像大小以提升处理速度
    original_height, original_width = rgb_array.shape[:2]
    if (PARAMS.get("TARGET_HEIGHT") and PARAMS.get("TARGET_WIDTH") and 
        (original_height > PARAMS["TARGET_HEIGHT"] or original_width > PARAMS["TARGET_WIDTH"])):
        
        rgb_array = cv2.resize(rgb_array, (PARAMS["TARGET_WIDTH"], PARAMS["TARGET_HEIGHT"]))
        logging.info(f"Image resized from {original_width}x{original_height} to {PARAMS['TARGET_WIDTH']}x{PARAMS['TARGET_HEIGHT']}")
    
    logging.info(f"Input Image: {img_path}")
    logging.info(f"Image Size: {rgb_array.shape[1]}x{rgb_array.shape[0]}")
    return rgb_array

def estimate_depth_optimized(generator, rgb_array):
    """优化的深度估计（模型已预热）"""
    with profiler.timer("🧠 Depth_estimation"):
        depth = generator.model.infer_image(rgb_array)
        return cp.asarray(depth, dtype=cp.float32)

def save_results_optimized(left, right, left_repaired, right_repaired, depth, output_dir):
    """优化的结果保存"""
    # 并行保存多个图像
    def save_image(img, path, is_gpu=True):
        if is_gpu and hasattr(img, 'get'):
            img = img.get()
        elif is_gpu:
            img = cp.asnumpy(img)
        
        if len(img.shape) == 3 and img.shape[2] == 3:  # RGB图像
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), img_bgr)
        else:
            cv2.imwrite(str(path), img)
    
    output_path = Path(output_dir)
    
    # 保存主要图像
    save_image(left, output_path / "left.png")
    save_image(right, output_path / "right.png")
    save_image(left_repaired, output_path / "left_repaired.png", is_gpu=False)
    save_image(right_repaired, output_path / "right_repaired.png", is_gpu=False)
    
    # 修复深度图保存，消除警告
    try:
        # 正确获取深度数据
        if hasattr(depth, 'get'):
            depth_cpu = depth.get()
        elif isinstance(depth, cp.ndarray):
            depth_cpu = cp.asnumpy(depth)
        else:
            depth_cpu = depth
        
        # 确保为numpy数组
        if not isinstance(depth_cpu, np.ndarray):
            depth_cpu = np.array(depth_cpu)
        
        # 正确处理深度图数据类型和范围
        if depth_cpu.dtype != np.uint8:
            # 获取有效深度值（排除inf和nan）
            valid_mask = np.isfinite(depth_cpu)
            if np.any(valid_mask):
                depth_min = np.min(depth_cpu[valid_mask])
                depth_max = np.max(depth_cpu[valid_mask])
                
                # 归一化到0-255范围
                if depth_max > depth_min:
                    depth_normalized = np.zeros_like(depth_cpu, dtype=np.uint8)
                    depth_normalized[valid_mask] = ((depth_cpu[valid_mask] - depth_min) / 
                                                   (depth_max - depth_min) * 255).astype(np.uint8)
                else:
                    depth_normalized = np.zeros_like(depth_cpu, dtype=np.uint8)
            else:
                depth_normalized = np.zeros_like(depth_cpu, dtype=np.uint8)
        else:
            depth_normalized = depth_cpu.astype(np.uint8)
        
        # 保存深度图（现在是正确的uint8格式）
        cv2.imwrite(str(output_path / 'depth.png'), depth_normalized)
        
        if 'valid_mask' in locals() and np.any(valid_mask):
            logging.info(f"Depth map saved (range: {depth_min:.3f}-{depth_max:.3f}m → 0-255)")
        else:
            logging.info("Depth map saved (uint8 format)")
            
    except Exception as e:
        logging.warning(f"Depth map save failed: {e}")

def post_process_optimized(output_dir, width, height):
    """优化的后处理（可选红青3D功能）"""
    if not PARAMS["ENABLE_POST_PROCESS"]:
        logging.info("Post-processing disabled for speed")
        return
        
    output_dir = Path(output_dir).absolute()
    try:
        # 可选：生成红青3D图
        if PARAMS["ENABLE_RED_CYAN"]:
            subprocess.run([
                "StereoscoPy",
                "-S", "5", "0",
                "-a",
                "-m", "color",
                "--cs", "red-cyan",
                "--lc", "rgb",
                str(output_dir / "left_repaired.png"),
                str(output_dir / "right_repaired.png"),
                str(output_dir / "red_cyan.jpg")
            ], check=True, capture_output=True)
            logging.info("Red-cyan 3D image generated")
        else:
            logging.info("Red-cyan 3D generation skipped (disabled for speed)")

        # 生成左右拼接立体图
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",  # 减少日志输出
            "-i", str(output_dir / "left_repaired.png"),
            "-i", str(output_dir / "right_repaired.png"),
            "-filter_complex", f"[0:v][1:v]hstack", 
            "-frames:v", "1", 
            str(output_dir / "stereo.jpg")
        ], check=True, capture_output=True)

        # 可选：生成视频
        if PARAMS["GENERATE_VIDEO"]:
            logging.info("Generating video...")
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-loop", "1",
                "-i", str(output_dir / "stereo.jpg"),
                "-c:v", "libx264",
                "-t", "60",
                "-pix_fmt", "yuv420p",
                "-vf", "fps=30",
                str(output_dir / "stereo_output.mp4")
            ], check=True, capture_output=True)

    except subprocess.CalledProcessError as e:
        logging.error(f"Post-process failure: {e.returncode}")
    except Exception as e:
        logging.error(f"Post-process error: {str(e)}")
    else:
        red_cyan_status = "with red-cyan 3D" if PARAMS["ENABLE_RED_CYAN"] else "without red-cyan 3D"
        logging.info(f"Post-processing completed ({red_cyan_status})")
    
    logging.info(f"Output Dir: {output_dir}")

# =============== 主流程优化版 ===============
def main_optimized():
    """优化的主函数（专注核心计时）"""
    profiler.reset()
    
    # 配置参数（不参与核心计时）
    img_path = PARAMS["IMG_PATH"]
    IPD = PARAMS["IPD"]
    PRO_RADIUS = PARAMS["PRO_RADIUS"]
    PIXEL = PARAMS["PIXEL"]
    CRITICAL_DEPTH = min(PARAMS["CRITICAL_DEPTH"], cal_critical_depth(IPD, PIXEL))
    
    logging.info(f"\n{'='*60}")
    logging.info(f"🎯 Core Processing Performance Analysis")
    logging.info(f"{'='*60}")
    
    try:
        # 1. 📸 图像加载（核心计时开始）
        with profiler.timer("📸 Image_loading"):
            rgb_array = load_and_resize_image(img_path)
            height, width = rgb_array.shape[:2]
            logging.info(f"Image loaded: {width}x{height}")

        # 2. 生成器初始化（含模型预热，不参与核心计时）
        logging.info(f'[Optimized Stereo] IPD: {IPD}m, Critical Depth: {CRITICAL_DEPTH:.2f}m')
        # 强制刷新日志
        for handler in logging.getLogger().handlers:
            handler.flush()
            
        generator = OptimizedStereoPano(
            height=height, width=width, IPD=IPD, pro_radius=PRO_RADIUS, 
            pixel=PIXEL, critical_depth=CRITICAL_DEPTH, 
            vit_size=PARAMS["VIT_SIZE"], dataset=PARAMS["DATASET"]
        )
        
        # 3. 🧠 深度估计（核心计时）
        with profiler.timer("🧠 Depth_estimation"):
            depth = generator.model.infer_image(rgb_array)
            depth_gpu = cp.asarray(depth, dtype=cp.float32)
            d_rgb = cp.asarray(rgb_array, dtype=cp.float32)
        
        # 4. 🎯 立体对生成（核心计时 - 使用原版本方法）
        with profiler.timer("🎯 Stereo_pair_generation"):
            left, right = generator.generate_stereo_pair_vectorized(d_rgb, depth_gpu)
        
        # 5. 🔧 图像修复（核心计时 - 并行处理）
        with profiler.timer("🔧 Image_repair"):
            # 并行修复左右图像
            import concurrent.futures
            
            def repair_single_image(image):
                """单个图像修复函数"""
                if not PARAMS["ENABLE_REPAIR"]:
                    return cp.asnumpy(image)
                    
                # 转换到CPU进行修复
                image_cpu = cp.asnumpy(image).astype(np.uint8)
                
                # 快速检测是否需要修复
                gray = cv2.cvtColor(image_cpu, cv2.COLOR_RGB2GRAY)
                black_pixels = np.sum(gray == 0)
                
                if black_pixels < gray.size * 0.01:  # 如果黑色像素少于1%，跳过修复
                    return image_cpu
                
                # 使用更快的修复算法
                img_bgr = cv2.cvtColor(image_cpu, cv2.COLOR_RGB2BGR)
                mask = np.uint8(gray == 0) * 255
                repaired = cv2.inpaint(img_bgr, mask, inpaintRadius=2, flags=cv2.INPAINT_NS)
                return cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)
            
            # 使用线程池并行处理左右图像
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # 提交左右图像修复任务
                future_left = executor.submit(repair_single_image, left)
                future_right = executor.submit(repair_single_image, right)
                
                # 获取结果
                left_repaired = future_left.result()
                right_repaired = future_right.result()

        # 保存和后处理（不参与核心计时）
        output_dir = generator.make_output_dir()
        save_results_optimized(left, right, left_repaired, right_repaired, depth_gpu, output_dir)
        post_process_optimized(output_dir, width, height)
        
        # 核心处理性能报告
        core_steps = ["📸 Image_loading", "🧠 Depth_estimation", "🎯 Stereo_pair_generation", "🔧 Image_repair"]
        total_core_time = sum(profiler.get_timing(step) for step in core_steps if profiler.get_timing(step))
        
        logging.info(f"\n🏆 Core Processing Summary:")
        for step in core_steps:
            time_val = profiler.get_timing(step)
            if time_val:
                logging.info(f"  {step}: {time_val:.3f}s")
        logging.info(f"  💫 Total Core Time: {total_core_time:.3f}s")
        
        # 强制刷新所有日志
        for handler in logging.getLogger().handlers:
            handler.flush()
        
        # 内存清理
        del d_rgb, depth_gpu, left, right
        cp.get_default_memory_pool().free_all_blocks()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logging.info("=== Processing completed successfully ===")
        # 最终刷新
        for handler in logging.getLogger().handlers:
            handler.flush()
            
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    logging_setup()
    main_optimized()
