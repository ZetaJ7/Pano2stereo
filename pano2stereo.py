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

# ================== 参数集中管理 ==================
PARAMS = {
    "IPD": 0.032,                      # 视环半径(m)
    "PRO_RADIUS": 1,                   # 视环投影半径(m)
    "PIXEL": 4448,                     # 赤道图像宽度(pixel)
    "CRITICAL_DEPTH": 9999,            # 临界深度(m)
    "GENERATE_VIDEO": 0,               # 是否生成视频
    "IMG_PATH": "Data/Lab3.jpg",       # 输入图片路径
    "VIT_SIZE": "vits",                # 深度模型类型
    "DATASET": "hypersim",             # 深度模型数据集
    "OUTPUT_BASE": "results/output"    # 输出目录基础名
}

# ================== 性能监控 ==================
class PerformanceProfiler:
    def __init__(self):
        self.timings = {}
        self.active_timers = {}
    
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
            logging.info(f"[TIMER] {name}: {elapsed:.4f}s")
    
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
    file_handler = logging.FileHandler('logs/pano2stereo.log', mode='w')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            file_handler, logging.StreamHandler()
        ]
    )
    logging.info("Logging setup complete.")

def cal_critical_depth(r, pixel):
    return r / (math.sin(2 * math.pi / pixel))

class StereoPano():
    def __init__(self, height, width, IPD, pro_radius, pixel, critical_depth, vit_size='vits', dataset='hypersim'):
        logging.info('Initializing StereoPano Generator')
        with profiler.timer("StereoPano_init"):
            self.IPD = IPD
            self.width = width
            self.height = height
            self.pro_radius = pro_radius
            self.pixel = pixel
            self.critical_depth = critical_depth
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            
            with profiler.timer("Model_loading"):
                self.model = DepthAnythingV2(**{**model_configs[vit_size], 'max_depth': 20})
                self.model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{vit_size}.pth', map_location=self.device))
                self.model = self.model.to(self.device).eval()
            
            with profiler.timer("Sphere_coords_precompute"):
                self.sphere_coords = self.image_coords_to_sphere(width, height)
            
            # 激活模型
            with profiler.timer("Model_warmup"):
                random = np.array(np.random.rand(1, 3, height, width), dtype=np.float32)
                self.model.infer_image(random)
            
            logging.info('[DepthAnythingV2] Model loaded: {}-{}'.format(vit_size, dataset))

    def image_coords_to_sphere(self, image_width, image_height, x=None, y=None):
        with profiler.timer("Image_to_sphere_coords"):
            if x is None or y is None:
                x = cp.arange(image_width)
                y = cp.arange(image_height)
                xx, yy = cp.meshgrid(x, y)
            else:
                xx, yy = cp.array(x), cp.array(y)
            u = (xx / (image_width - 1)) * 2 - 1
            v = (yy / (image_height - 1)) * -2 + 1
            lon = u * cp.pi
            lat = v * (cp.pi / 2)
            x_xyz = cp.cos(lat) * cp.cos(lon)
            y_xyz = cp.cos(lat) * cp.sin(lon)
            z_xyz = cp.sin(lat)
            r = cp.sqrt(x_xyz ** 2 + y_xyz ** 2 + z_xyz ** 2)
            safe_r = cp.where(r == 0, 1e-12, r)
            theta = cp.arccos(z_xyz / safe_r)
            phi = cp.arctan2(y_xyz, x_xyz)
            theta = cp.where(r == 0, 0.0, theta)
            phi = cp.where(r == 0, 0.0, phi)
            sphere_coords = cp.stack([r, theta, phi], axis=-1)
            return sphere_coords.squeeze()

    def sphere_to_image_coords(self, sphere_coords, image_width, image_height):
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

    def normalize_channel(self, data, percentile=99):
        data = cp.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        max_val = cp.percentile(data, percentile)
        min_val = cp.percentile(data, 100 - percentile)
        data = cp.clip(data, min_val, max_val)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        return normalized

    def repair_black_regions(self, image):
        with profiler.timer("Repair_black_regions"):
            image = cp.asnumpy(image).astype(np.uint8)
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            mask = np.uint8(gray == 0) * 255
            repaired = cv2.inpaint(img_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
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

    def save_depth_map(self, depth, output_path):
        depth = depth.get()
        min_val, max_val = depth.min(), depth.max()
        logging.info(f'Depth min: {min_val}, max: {max_val}')
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_normalized.astype(np.uint8)
        cv2.imwrite(str(Path(output_path) / 'depth.png'), depth_uint8)

    def sphere2pano_vectorized(self, sphere_coords, z_vals):
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
        with profiler.timer("Generate_stereo_pair_total"):
            logging.info("=============== Generating stereo pair ================")
            
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
        return self.generate_stereo_pair_vectorized(rgb_array, depth)

# =============== 功能函数 ===============

def load_image(img_path):
    with profiler.timer("Load_image"):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File Not Found: {img_path}")
        bgr_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr_array is None:
            raise FileNotFoundError(f"Cannot Read file: {img_path}")
        rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
        logging.info(f"Input Image: {img_path}")
        logging.info(f"Image Size: {rgb_array.shape[1]}x{rgb_array.shape[0]}")
        return rgb_array

def estimate_depth(model, rgb_array):
    with profiler.timer("Depth_estimation"):
        depth = model.infer_image(rgb_array)
        return cp.asarray(depth, dtype=cp.float32)

def save_results(left, right, left_repaired, right_repaired, depth, output_dir, width, height):
    with profiler.timer("Save_results"):
        left = cp.asnumpy(left)
        right = cp.asnumpy(right)
        left_bgr = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
        right_bgr = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(Path(output_dir) / "left.png"), left_bgr)
        cv2.imwrite(str(Path(output_dir) / "right.png"), right_bgr)
        left_repaired = cp.asnumpy(left_repaired)
        right_repaired = cp.asnumpy(right_repaired)
        cv2.imwrite(str(Path(output_dir) / "left_repaired.png"), cv2.cvtColor(left_repaired, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(Path(output_dir) / "right_repaired.png"), cv2.cvtColor(right_repaired, cv2.COLOR_RGB2BGR))

        # 保存深度图
        try:
            with profiler.timer("Save_depth_map"):
                depth_cpu = depth.get() if hasattr(depth, 'get') else depth
                min_val, max_val = depth_cpu.min(), depth_cpu.max()
                logging.info(f'Depth min: {min_val}, max: {max_val}')
                depth_normalized = cv2.normalize(depth_cpu, None, 0, 255, cv2.NORM_MINMAX)
                depth_uint8 = depth_normalized.astype(np.uint8)
                cv2.imwrite(str(Path(output_dir) / 'depth.png'), depth_uint8)
        except Exception as e:
            logging.warning(f"Depth map save failed: {e}")

def post_process(output_dir, width, height):
    with profiler.timer("Post_process"):
        output_dir = Path(output_dir).absolute()
        try:
            # 红青3D
            with profiler.timer("Generate_red_cyan_3D"):
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
                ], check=True)

            # 左右拼接立体图
            with profiler.timer("Generate_stereo_pair_image"):
                subprocess.run([
                    "ffmpeg",
                    "-y",
                    "-i", str(output_dir / "left_repaired.png"),
                    "-i", str(output_dir / "right_repaired.png"),
                    "-filter_complex",
                    f"[0:v]scale={int(width)}:{int(height)}[img1];[img1][1:v]hstack","-frames:v", "1", str(output_dir / "stereo.jpg")
                ], check=True)

            if PARAMS["GENERATE_VIDEO"]:
                logging.info("Generating video...")
                with profiler.timer("Generate_video"):
                    subprocess.run([
                        "ffmpeg",
                        "-y",
                        "-loop", "1",
                        "-i", str(output_dir / "stereo.jpg"),
                        "-c:v", "libx264",
                        "-t", "60",
                        "-pix_fmt", "yuv420p",
                        "-vf", "fps=30",
                        str(output_dir / "stereo_output.mp4")
                    ], check=True)

        except subprocess.CalledProcessError as e:
            logging.error(f"Failure: {e.returncode}\n{e.stderr}")
        except Exception as e:
            logging.error(f"Error: {str(e)}")
        else:
            logging.info("Panoramic to Stereo conversion completed successfully.")
        logging.info(f"Output Dir: {output_dir}")

# =============== 主流程 ===============
def main():
    with profiler.timer("Total_execution_time"):
        # 1. 加载图片
        with profiler.timer("Image_loading_and_setup"):
            img_path = PARAMS["IMG_PATH"]
            rgb_array = load_image(img_path)
            height, width = rgb_array.shape[:2]
            d_rgb = cp.asarray(rgb_array, dtype=cp.float32)

        # 2. 参数计算
        with profiler.timer("Parameter_calculation"):
            IPD = PARAMS["IPD"]
            PRO_RADIUS = PARAMS["PRO_RADIUS"]
            PIXEL = PARAMS["PIXEL"]
            CRITICAL_DEPTH = min(PARAMS["CRITICAL_DEPTH"], cal_critical_depth(IPD, PIXEL))
            logging.info('[Stereo Parameters]\nCircular Radius: %sm\nProjection Radius: %sm\nPixel on Equator: %s\nCritical Depth: %sm\n===============================' % (IPD, PRO_RADIUS, PIXEL, CRITICAL_DEPTH))

        # 3. 初始化生成器与深度估计
        with profiler.timer("Generator_initialization"):
            generator = StereoPano(height=height, width=width, IPD=IPD, pro_radius=PRO_RADIUS, pixel=PIXEL,
                                   critical_depth=CRITICAL_DEPTH, vit_size=PARAMS["VIT_SIZE"], dataset=PARAMS["DATASET"])
        
        depth = estimate_depth(generator.model, rgb_array)
        assert d_rgb.shape[:2] == depth.shape[:2]

        # 4. 生成立体对
        left, right = generator.generate_stereo_pair(d_rgb, depth)
        
        with profiler.timer("Image_repair"):
            left_repaired = generator.repair_black_regions(left)
            right_repaired = generator.repair_black_regions(right)

        # 5. 保存结果
        with profiler.timer("Output_directory_creation"):
            output_dir = generator.make_output_dir()
        
        save_results(left, right, left_repaired, right_repaired, depth, output_dir, width, height)

        # 6. 后处理（红青3D、左右拼接、可选视频）
        post_process(output_dir, width, height)
    
    # 输出性能统计
    logging.info(profiler.get_summary())

if __name__ == "__main__":
    logging_setup()
    main()






