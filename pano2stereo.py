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
from submodule.Flashdepth.inference import FlashDepthProcessor
import cupy as cp
from contextlib import contextmanager
import gc
import concurrent.futures  
from PIL import Image
import stereoscopy

# Early logging configuration so module-level logging (before logging_setup())
# is captured to both console and the log file. This prevents messages emitted
# during module import from being lost to the console only.
try:
    os.makedirs('logs', exist_ok=True)
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('logs/pano2stereo.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def cal_critical_depth(r, pixel):
    return r / (math.sin(2 * math.pi / pixel))

def torch_to_cupy(tensor1, tensor2):
    """Convert two torch tensors to CuPy arrays"""
    # Minimal, fast conversion with zero-copy on CUDA using DLPack when possible.
    # If the tensor is on CUDA, use DLPack to avoid CPU<->GPU copy. Otherwise fallback to cpu->numpy->cupy.
    tensor1 = tensor1.float()
    tensor2 = tensor2.float()
    
    from torch.utils import dlpack as _dlpack

    def _to_cupy(x):
        if isinstance(x, torch.Tensor):
            if x.is_cuda:
                # Ensure contiguous to avoid unexpected behavior
                if not x.is_contiguous():
                    x = x.contiguous()
                try:
                    return cp.fromDlpack(_dlpack.to_dlpack(x))
                except Exception:
                    # Fallback if DLPack path fails for any reason
                    return cp.asarray(x.detach().cpu().numpy())
            else:
                return cp.asarray(x.detach().cpu().numpy())
        else:
            return cp.asarray(x)

    return _to_cupy(tensor1), _to_cupy(tensor2)

# Load parameters from YAML config file
def load_config(file):
    """Load parameters from YAML config file"""
    config_file_yaml = file
    
    # Load YAML configuration
    if os.path.exists(config_file_yaml):
        try:
            import yaml
            with open(config_file_yaml, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            
            # Convert YAML parameters to code format
            params = {
                "URL": yaml_config.get("URL", None), 
                "IPD": yaml_config.get("ipd", 0.032),
                "PRO_RADIUS": yaml_config.get("pro_radius", 1),
                "PIXEL": yaml_config.get("pixel", 4448),
                "CRITICAL_DEPTH": yaml_config.get("critical_depth", 9999),
                "GENERATE_VIDEO": yaml_config.get("generate_video", False),
                "IMG_PATH": yaml_config.get("img_path", "Data/Lab1_2k.png"),
                "VIT_SIZE": yaml_config.get("vit_size", "vits"),
                "DATASET": yaml_config.get("dataset", "hypersim"),
                "OUTPUT_BASE": yaml_config.get("output_base", "results/output"),
                "ENABLE_REPAIR": yaml_config.get("enable_repair", True),
                "ENABLE_POST_PROCESS": yaml_config.get("enable_post_process", True),
                "ENABLE_RED_CYAN": yaml_config.get("enable_red_cyan", False),
                "TARGET_HEIGHT": yaml_config.get("target_height", 540),
                "TARGET_WIDTH": yaml_config.get("target_width", 960),
                # FlashDepth related configuration
                "FLASHDEPTH_CONFIG": yaml_config.get("flashdepth", {})
            }
            logging.info("Configuration loaded from config.yaml")
            return params
            
        except ImportError:
            logging.error("PyYAML not installed. Please install PyYAML: pip install PyYAML")
            raise
        except Exception as e:
            logging.error(f"Failed to load YAML config: {e}")
            raise
    
    # If YAML file does not exist, use default parameters
    logging.warning("yaml not found, using default parameters")
    return {
        "URL": None,  # Default URL
        "IPD": 0.032,                      # Pupil distance (m)
        "PRO_RADIUS": 1,                   # Projection radius (m)
        "PIXEL": 4448,                     # Equatorial image width (pixel)
        "CRITICAL_DEPTH": 9999,            # Critical depth (m)
        "GENERATE_VIDEO": False,           # Whether to generate video
        "IMG_PATH": "Data/Lab1_2k.png",    # Input image path
        "VIT_SIZE": "vits",                # Depth model type
        "DATASET": "hypersim",             # Depth model dataset
        "OUTPUT_BASE": "results/output",   # Output directory base name
        "ENABLE_REPAIR": True,             # Whether to enable image repair
        "ENABLE_POST_PROCESS": True,       # Whether to enable post-processing
        "ENABLE_RED_CYAN": False,          # Whether to generate red-cyan 3D image
        "TARGET_HEIGHT": 540,              # Target height
        "TARGET_WIDTH": 960,               # Target width
        "FLASHDEPTH_CONFIG": {}
    }

# Load configuration parameters
PARAMS = load_config('configs/pano.yaml')
# logging.info(f"Loaded parameters: {PARAMS}")

# ================== Performance Monitoring ==================
class PerformanceProfiler:
    def __init__(self):
        self.timings = {}
        self.active_timers = {}
    
    def reset(self):
        """Reset timer data"""
        self.timings.clear()
        self.active_timers.clear()
    
    def get_timing(self, name):
        """Get the latest time for the specified timer"""
        if name in self.timings and self.timings[name]:
            return self.timings[name][-1]  # Return the latest time
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
            
            # Output immediately and flush log
            message = f"[TIMER] {name}: {elapsed:.4f}s"
            logging.info(message)
            # Force flush all handlers
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

# Global performance profiler
profiler = PerformanceProfiler()

def logging_setup():
    # Set OpenCV log level to reduce warnings
    try:
        cv2.setLogLevel(3)  # 3=ERROR level, reduce warning messages
    except:
        pass  # Skip if setting fails
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging - force flush buffer
    file_handler = logging.FileHandler('logs/pano2stereo.log', mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Include module and line number to help quickly locate the source of messages
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Get root logger and configure
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Clear existing handlers and install our handlers
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Ensure other existing loggers (from submodules) propagate to root so
    # their messages are captured by the root handlers. Some libraries add
    # their own handlers; remove them to centralize logging to our file.
    try:
        for name, logger_obj in list(logging.root.manager.loggerDict.items()):
            # Skip internal entries that are not Logger instances
            if not isinstance(logger_obj, logging.Logger):
                continue
            # Remove any handlers on the logger and let it propagate to root
            try:
                logger_obj.handlers.clear()
                logger_obj.propagate = True
            except Exception:
                pass
    except Exception:
        # Non-fatal; best-effort
        pass
    
    # Force flush
    logging.info("=== Pano2Stereo Optimized Processing Started ===")
    for handler in root_logger.handlers:
        handler.flush()

def get_stream_resolution(url):
    """Get resolution from video stream"""
    try:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video stream: {url}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        cap.release()
        
        logging.info(f"Stream resolution: {width}x{height}, FPS: {fps}")
        return width, height, fps
    except Exception as e:
        logging.error(f"Failed to get stream resolution: {e}")
        raise

class Pano2stereo():
    def __init__(self, IPD, pro_radius, pixel, critical_depth, url):
        logging.info('Initializing Pano2stereo Generator...')
        self.IPD = IPD
        self.width = None
        self.height = None
        self.pro_radius = pro_radius
        self.pixel = pixel
        self.critical_depth = critical_depth
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.outputdir = self.make_output_dir()
        self.url = url
        self.target_url = 'rtsp://10.20.35.30:28552/result'
        self.running = True
        self.save_result = False
        
        # Initialize RTSP streaming
        self.stream_process = None
        self.stream_fps = 30  # Default FPS for streaming
        # Note: RTSP streaming will be initialized after getting dimensions from FlashDepth
        
        # Load model
        logging.info("Initializing FlashDepth model...")
        self.flashdepth_processor = FlashDepthProcessor(config_path="configs/flashdepth.yaml", url=self.url, stream_mode=True, save_depth_png=False, save_frame=False, max_frames=20, run_dir=self.outputdir)
        
        # Start FlashDepth inference in a separate thread
        self.inference_thread = threading.Thread(target=self.flashdepth_processor.run_inference, daemon=True)
        self.inference_thread.start()
        logging.info('[FlashDepth] Inference thread started in Pano2stereo.__init__')
        
        # Catch width and height from FlashdepthProcessor
        logging.info("Waiting for FlashDepth inference to start...")
        # Wait for FlashDepth processor to initialize pred
        while self.flashdepth_processor.pred is None:
            time.sleep(0.02)  # Wait 20ms for inference to start
                
        # self.pred = [depth_pred, original_frame]  # Store latest depth map & frame（Tensor, shape [H, W, C] in BGR)
        self.height = self.flashdepth_processor.pred[0].shape[0]
        self.width = self.flashdepth_processor.pred[0].shape[1]
        logging.info(f"Got Input resolution from FlashDepth: {self.width}x{self.height}")
        
        # Now initialize RTSP streaming with correct dimensions
        self._init_streaming()

        # Precompute all required coordinate transformations
        logging.info("Precomputing sphere coordinates...")
        self.sphere_coords = self._precompute_sphere_coords(self.width, self.height)  # (r, theta, phi) for [H,W] resolution, shape [H, W, 3]

        logging.info('[Pano2stereo] Initialization completed')



    def _init_streaming(self):
        """Initialize RTSP streaming pipeline"""
        try:
            # Calculate output resolution (left + right images side by side)
            output_width = self.width * 2
            output_height = self.height
            
            # ffmpeg command for RTSP streaming
            # Low-latency friendly ffmpeg options; keep rawvideo RGB24 input from stdin
            ffmpeg_cmd = [
                'ffmpeg',
                '-loglevel', 'debug',
                '-y',
                '-f', 'rawvideo',
                '-pix_fmt', 'rgb24',
                '-s', f'{output_width}x{output_height}',
                '-r', str(self.stream_fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-g', '30',
                '-fflags', 'nobuffer',
                '-flags', 'low_delay',
                '-max_delay', '0',
                '-flush_packets', '1',
                '-f', 'rtsp',
                '-rtsp_transport', 'tcp',
                self.target_url
            ]
            
            logging.info(f"Initializing RTSP stream to {self.target_url}")
            logging.info(f"Stream resolution: {output_width}x{output_height} @ {self.stream_fps}fps")

            # Start ffmpeg process and capture stderr for diagnostics
            # Prepare per-run stream directory and ffmpeg debug log file
            try:
                stream_dir = Path(self.outputdir) / 'stream'
                stream_dir.mkdir(parents=True, exist_ok=True)
                ffmpeg_log_path = stream_dir / 'ffmpeg_debug.log'
                # Open in append-binary mode so ffmpeg bytes can be written directly
                self._ffmpeg_log_file = open(str(ffmpeg_log_path), 'ab')
            except Exception as e:
                logging.warning(f"Failed to open ffmpeg debug log file: {e}; falling back to DEVNULL")
                self._ffmpeg_log_file = None

            # Launch ffmpeg; write stderr to per-run debug file when available
            stderr_target = self._ffmpeg_log_file if self._ffmpeg_log_file is not None else subprocess.DEVNULL
            self.stream_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=stderr_target,
                bufsize=0
            )

            if self._ffmpeg_log_file:
                logging.info(f"ffmpeg debug log: {ffmpeg_log_path}")

            # Short delay and check process health
            time.sleep(0.2)
            if self.stream_process.poll() is not None:
                logging.error(f"ffmpeg exited immediately with code {self.stream_process.returncode}")
                # Let caller know stream wasn't initialized
                raise RuntimeError("ffmpeg failed to start streaming; check ffmpeg stderr in logs")

            logging.info("RTSP streaming initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize RTSP streaming: {e}")
            self.stream_process = None

    def stream_frame(self, frame):
        """Stream a frame to RTSP"""
        if self.stream_process and self.stream_process.poll() is None:
            try:
                # Ensure frame is in correct format (RGB, uint8)
                if isinstance(frame, cp.ndarray):
                    frame = cp.asnumpy(frame)
                # Ensure correct dtype and contiguous layout
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                frame = np.ascontiguousarray(frame)

                # Write frame to ffmpeg stdin
                try:
                    # Validate expected frame size: height x (width*2) x 3
                    expected_h = getattr(self, 'height', None)
                    expected_w = getattr(self, 'width', None)
                    if expected_h is not None and expected_w is not None:
                            expected_shape = (expected_h, expected_w * 2, 3)
                            if frame.shape != expected_shape:
                                logging.warning(f"Frame shape mismatch: got {frame.shape}, expected {expected_shape}.")
                                # Try to derive new dimensions from incoming frame
                                new_h, new_w_total = frame.shape[0], frame.shape[1]
                                # If width is even, assume side-by-side stereo (two halves)
                                if new_w_total % 2 == 0:
                                    new_w = new_w_total // 2
                                else:
                                    new_w = None

                                # If difference is small, just resize to expected shape and continue streaming
                                if new_w is not None and abs(new_h - expected_h) <= 8 and abs(new_w - expected_w) <= 8:
                                    logging.info(f"Resizing minor-mismatch frame {frame.shape} -> {expected_shape}")
                                    try:
                                        frame = cv2.resize(frame, (expected_w * 2, expected_h), interpolation=cv2.INTER_LINEAR)
                                    except Exception as e:
                                        logging.error(f"Failed to resize frame: {e}. Skipping frame.")
                                        return
                                else:
                                    # Significant change: resize frame to match current stream resolution
                                    if new_w is not None:
                                        logging.info(f"Resizing significant-mismatch frame {frame.shape} -> {expected_shape}")
                                        try:
                                            frame = cv2.resize(frame, (expected_w * 2, expected_h), interpolation=cv2.INTER_LINEAR)
                                        except Exception as e:
                                            logging.error(f"Failed to resize frame: {e}. Skipping frame.")
                                            return
                                    else:
                                        logging.error("Incoming frame width is odd; cannot split into two views. Skipping frame.")
                                        return
                    data = frame.tobytes()
                    logging.debug(f"Writing frame to ffmpeg stdin: bytes={len(data)}, expected={frame.size}")
                    self.stream_process.stdin.write(data)
                    self.stream_process.stdin.flush()
                except BrokenPipeError:
                    logging.warning("ffmpeg stdin broken (BrokenPipeError). Restarting stream process")
                    self._restart_streaming()
                except Exception as e:
                    logging.error(f"Failed to write frame to ffmpeg stdin: {e}")
                    self._restart_streaming()
                
            except Exception as e:
                logging.error(f"Failed to stream frame: {e}")
                # Try to restart streaming
                self._restart_streaming()
        else:
            logging.warning("Stream process not available, attempting to restart...")
            self._restart_streaming()

    def _restart_streaming(self):
        """Restart RTSP streaming"""
        if self.stream_process:
            try:
                self.stream_process.terminate()
                self.stream_process.wait(timeout=5)
            except:
                self.stream_process.kill()
        # Close ffmpeg log file handle if open (we'll reopen in _init_streaming)
        try:
            if hasattr(self, '_ffmpeg_log_file') and self._ffmpeg_log_file is not None:
                try:
                    self._ffmpeg_log_file.flush()
                except Exception:
                    pass
                try:
                    self._ffmpeg_log_file.close()
                except Exception:
                    pass
                self._ffmpeg_log_file = None
        except Exception:
            pass

        self._init_streaming()

    def stop_streaming(self):
        """Stop RTSP streaming"""
        if self.stream_process:
            try:
                self.stream_process.stdin.close()
                self.stream_process.terminate()
                self.stream_process.wait(timeout=5)
                logging.info("RTSP streaming stopped")
            except Exception as e:
                logging.error(f"Error stopping stream: {e}")
                try:
                    self.stream_process.kill()
                except:
                    pass
        # Close ffmpeg debug log file if open
        try:
            if hasattr(self, '_ffmpeg_log_file') and self._ffmpeg_log_file is not None:
                try:
                    self._ffmpeg_log_file.flush()
                except Exception:
                    pass
                try:
                    self._ffmpeg_log_file.close()
                except Exception:
                    pass
                self._ffmpeg_log_file = None
        except Exception:
            pass

    def _precompute_sphere_coords(self, image_width, image_height):
        """get spherical coordinates (r, theta, phi) for all pixels in (H,W) resolution, return shape [H, W, 3]"""
        # Use more efficient coordinate generation
        x = cp.linspace(0, image_width - 1, image_width, dtype=cp.float32)
        y = cp.linspace(0, image_height - 1, image_height, dtype=cp.float32)
        xx, yy = cp.meshgrid(x, y)
        
        # Vectorized calculation of spherical coordinates
        u = (xx / (image_width - 1)) * 2 - 1
        v = (yy / (image_height - 1)) * -2 + 1
        lon = u * cp.pi
        lat = v * (cp.pi / 2)
        
        # Unit sphere to Cartesian coordinates
        x_xyz = cp.cos(lat) * cp.cos(lon)
        y_xyz = cp.cos(lat) * cp.sin(lon)
        z_xyz = cp.sin(lat)
        
        # Use more stable calculation
        r = cp.ones_like(x_xyz)  # Unit sphere, r=1
        theta = cp.arccos(cp.clip(z_xyz, -1, 1))
        phi = cp.arctan2(y_xyz, x_xyz)
        
        return cp.stack([r, theta, phi], axis=-1)

    def sphere2pano_vectorized(self, depth_map):
        """Original version of efficient spherical parallax calculation"""
        with profiler.timer("Sphere2pano_vectorized"):
            # Ensure depth_map is 2D
            depth_map = cp.asarray(depth_map)
            assert depth_map.ndim == 2, "depth_map must be 2D, got shape {}".format(depth_map.shape)
            
            R = cp.asarray(self.sphere_coords[..., 0])
            theta = cp.asarray(self.sphere_coords[..., 1])
            phi = cp.asarray(self.sphere_coords[..., 2])
            mask = depth_map < self.critical_depth
            ratio = cp.clip(self.IPD / cp.where(mask, depth_map, cp.inf), -1.0, 1.0)
            delta = cp.arcsin(ratio)
            phi_l = (phi + delta) % (2 * cp.pi)
            phi_r = (phi - delta) % (2 * cp.pi)
            return (
                cp.stack([R, theta, phi_l], axis=-1),
                cp.stack([R, theta, phi_r], axis=-1)
            )
    
    def sphere_to_image_coords(self, sphere_coords, image_width, image_height):
        """Modified for forward mapping: map sphere coords to target image coords"""
        with profiler.timer("Sphere_to_image_coords"):
            theta = sphere_coords[..., 1]
            phi_orig = sphere_coords[..., 2]
            phi = (phi_orig + cp.pi) % (2 * cp.pi) - cp.pi
            theta = cp.clip(theta, 0, cp.pi)
            lat = cp.pi / 2 - theta
            lon = phi
            u = lon / cp.pi
            v = lat / (cp.pi / 2)
            x_float = (u + 1) * 0.5 * (image_width - 1)
            y_float = (1 - v) * 0.5 * (image_height - 1)
            x = cp.round(x_float).astype(cp.int32)
            y = cp.round(y_float).astype(cp.int32)
            x = cp.clip(x, 0, image_width - 1)
            y = cp.clip(y, 0, image_height - 1)

            # Seam fix
            try:
                eps = 1e-6
                seam_mask = (x == 0) & (phi_orig > (cp.pi - eps))
                if seam_mask.any():
                    x = cp.where(seam_mask, image_width - 1, x)
            except Exception:
                pass

            # For forward mapping, these are target coords
            target_coords = cp.stack([x, y], axis=-1)
            return target_coords

    def generate_stereo_pair(self, rgb_data, depth_data):
        """Modified for forward mapping with depth-based z-buffering"""
        with profiler.timer("Generate_stereo_pair_total"):           
            with profiler.timer("Depth fix on sphere pixel"):
                sphere_l, sphere_r = self.sphere2pano_vectorized(depth_data)
            with profiler.timer("Revert sphere to image pixel"):
                coords_l = self.sphere_to_image_coords(sphere_l, self.width, self.height)
                coords_r = self.sphere_to_image_coords(sphere_r, self.width, self.height)
            with profiler.timer("Image_sampling"):
                # Delegate to helper method that performs forward mapping with z-buffer on GPU
                return self._forward_map_zbuffer(coords_l, coords_r, rgb_data, depth_data)

    def _forward_map_zbuffer(self, coords_l, coords_r, rgb_data, depth_data):
        """Forward-map source pixels to target using a single-pass GPU z-buffer kernel.

        Inputs:
        - coords_l/coords_r: CuPy arrays of shape (H,W,2) containing integer target coords
        - rgb_data: CuPy array (H,W,3) uint8
        - depth_data: CuPy array (H,W) float32

        Returns: left, right (CuPy arrays H,W,3) and stores unmapped masks in self._last_masks
        """
        # Prepare integer target coordinates (flattened)
        xl = coords_l[..., 0].ravel().astype(cp.int32)
        yl = coords_l[..., 1].ravel().astype(cp.int32)
        xr = coords_r[..., 0].ravel().astype(cp.int32)
        yr = coords_r[..., 1].ravel().astype(cp.int32)

        # Number of source pixels
        N = int(self.height * self.width)

        # Source flattened arrays
        src_y, src_x = cp.meshgrid(cp.arange(self.height, dtype=cp.int32), cp.arange(self.width, dtype=cp.int32), indexing='ij')
        src_y = src_y.ravel()
        src_x = src_x.ravel()
        src_depth = depth_data.ravel().astype(cp.float32)
        src_rgb = rgb_data.reshape(-1, 3).astype(cp.uint8).ravel()

        # Allocate target buffers (flattened)
        depth_buf_l = cp.full((self.height * self.width,), 1e30, dtype=cp.float32)
        rgb_buf_l = cp.zeros((self.height * self.width * 3,), dtype=cp.uint8)
        count_buf_l = cp.zeros((self.height * self.width,), dtype=cp.int32)

        depth_buf_r = cp.full((self.height * self.width,), 1e30, dtype=cp.float32)
        rgb_buf_r = cp.zeros((self.height * self.width * 3,), dtype=cp.uint8)
        count_buf_r = cp.zeros((self.height * self.width,), dtype=cp.int32)

        # CUDA kernel: atomic z-buffer scatter (uses atomicCAS on uint representation of floats)
        kernel_code = r'''
        extern "C" __global__
        void zbuffer_scatter(
            const int N,
            const int W,
            const int H,
            const int* tgt_x,
            const int* tgt_y,
            const float* depth_src,
            const unsigned char* rgb_src,
            float* depth_buf,
            unsigned char* rgb_buf,
            int* count_buf
        ) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= N) return;

            int x = tgt_x[i];
            int y = tgt_y[i];
            if (x < 0 || x >= W || y < 0 || y >= H) return;

            int idx = y * W + x;

            // increment count
            if (count_buf) atomicAdd(&count_buf[idx], 1);

            float d = depth_src[i];

            unsigned int* depth_ui = (unsigned int*)depth_buf;
            unsigned int old_ui = depth_ui[idx];
            float oldf = __uint_as_float(old_ui);
            unsigned int new_ui = __float_as_uint(d);

            while (d < oldf) {
                unsigned int prev = atomicCAS(depth_ui + idx, old_ui, new_ui);
                if (prev == old_ui) {
                    int rgb_idx = idx * 3;
                    rgb_buf[rgb_idx + 0] = rgb_src[i * 3 + 0];
                    rgb_buf[rgb_idx + 1] = rgb_src[i * 3 + 1];
                    rgb_buf[rgb_idx + 2] = rgb_src[i * 3 + 2];
                    break;
                }
                old_ui = prev;
                oldf = __uint_as_float(old_ui);
                if (!(d < oldf)) break;
            }
        }
        '''

        zbuffer_kernel = cp.RawKernel(kernel_code, 'zbuffer_scatter')

        threads = 256
        blocks = (N + threads - 1) // threads

        # Launch kernel for left and right (two separate launches)
        try:
            zbuffer_kernel((blocks,), (threads,), (
                N,
                self.width,
                self.height,
                xl, yl, src_depth, src_rgb, depth_buf_l, rgb_buf_l, count_buf_l
            ))
        except Exception as e:
            logging.error(f"zbuffer kernel (left) failed: {e}")

        try:
            zbuffer_kernel((blocks,), (threads,), (
                N,
                self.width,
                self.height,
                xr, yr, src_depth, src_rgb, depth_buf_r, rgb_buf_r, count_buf_r
            ))
        except Exception as e:
            logging.error(f"zbuffer kernel (right) failed: {e}")

        # Reshape outputs to H x W x 3
        left = rgb_buf_l.reshape((self.height, self.width, 3))
        right = rgb_buf_r.reshape((self.height, self.width, 3))

        # Masks for unmapped pixels (count==0)
        mask_left_unmapped = (count_buf_l.reshape((self.height, self.width)) == 0)
        mask_right_unmapped = (count_buf_r.reshape((self.height, self.width)) == 0)
        self._last_masks = (mask_left_unmapped, mask_right_unmapped)

        return left, right

    @staticmethod
    def repair_black_regions(image):
        """Fast black region repair (GPU-optimized)"""
        if not PARAMS["ENABLE_REPAIR"]:
            return cp.asnumpy(image)
        
        # Keep data on GPU as much as possible
        image_gpu = cp.asarray(image, dtype=cp.float32)
        
        # Convert to grayscale on GPU (RGB to grayscale: 0.299*R + 0.587*G + 0.114*B)
        if image_gpu.shape[-1] == 3:  # RGB image
            gray_gpu = cp.dot(image_gpu[..., :3], cp.array([0.299, 0.587, 0.114], dtype=cp.float32))
        else:
            gray_gpu = image_gpu
        
        # Count black pixels on GPU
        black_pixels = cp.sum(gray_gpu == 0)
        total_pixels = gray_gpu.size
        
        if black_pixels < total_pixels * 0.01:  # If black pixels less than 1%, skip repair
            return cp.asnumpy(image_gpu.astype(cp.uint8))
        
        # For repair, we need to move to CPU as cv2.inpaint doesn't have GPU support
        # But minimize the transfer by only transferring when repair is actually needed
        image_cpu = cp.asnumpy(image_gpu.astype(cp.uint8))
        
        # Use faster repair algorithm on CPU
        img_bgr = cv2.cvtColor(image_cpu, cv2.COLOR_RGB2BGR)
        mask = np.uint8(cv2.cvtColor(image_cpu, cv2.COLOR_RGB2GRAY) == 0) * 255
        repaired = cv2.inpaint(img_bgr, mask, inpaintRadius=2, flags=cv2.INPAINT_NS)
        return cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)

    def make_output_dir(self, base_dir='output'):
        """Create output directory under a parent folder as run_XXX.

        Examples:
        - default: output/run_000
        - with base_dir 'results/output': results/output/run_000
        """
        # Determine parent directory. If caller passed None, fall back to configured OUTPUT_BASE
        if base_dir is None:
            base_dir = PARAMS.get("OUTPUT_BASE", "output")

        parent_dir = Path(base_dir)
        # Ensure parent exists
        parent_dir.mkdir(parents=True, exist_ok=True)

        index = 0
        while True:
            target_dir = parent_dir / f"run_{index:03d}"
            if not target_dir.exists():
                target_dir.mkdir(parents=True, exist_ok=False)
                break
            index += 1

        logging.info(f"Output directory created: {target_dir}")
        return str(target_dir)

    def disparity_to_depth(self, disparity_map, critical_depth):
        """Convert disparity map to depth map and clamp to valid range.

        Each pixel x -> depth = 100.0 / x. Then negative depths are set to 0
        and values greater than `critical_depth` are clamped to `critical_depth`.

        Arguments:
            disparity_map: CuPy or NumPy array of disparities.
            critical_depth: scalar maximum depth (float).
        Returns:
            depth_map: array (same type as input) with depths in [0, critical_depth].
        """
        # Work with CuPy on GPU when available to avoid unnecessary copies
        try:
            if 'cp' in globals() and isinstance(disparity_map, cp.ndarray):
                # Avoid division by zero by flooring small disparities to a tiny positive value
                d = cp.clip(disparity_map, 1e-6, cp.inf)
                depth_map = 100.0 / d
                # Replace non-finite results with 0, clamp negatives to 0, and clamp to critical_depth
                depth_map = cp.where(cp.isfinite(depth_map), depth_map, 0.0)
                depth_map = cp.clip(depth_map, 0.0, float(critical_depth))
                return depth_map
        except Exception:
            # Fall through to numpy implementation on any CuPy failure
            pass

        # NumPy fallback (CPU)
        import numpy as _np
        d_np = _np.clip(_np.asarray(disparity_map), 1e-6, _np.inf)
        depth_map = 100.0 / d_np
        depth_map = _np.where(_np.isfinite(depth_map), depth_map, 0.0)
        depth_map = _np.clip(depth_map, 0.0, float(critical_depth))
        return depth_map
    
    def __del__(self):
        """Destructor to clean up resources"""
        logging.info('[Pano2stereo] Cleaning up resources...')
        
        # Stop running flag to signal threads to exit
        self.running = False
        
        # Stop FlashDepth inference thread
        if hasattr(self, 'inference_thread') and self.inference_thread and self.inference_thread.is_alive():
            logging.info('[Pano2stereo] Waiting for inference thread to finish...')
            self.inference_thread.join(timeout=5.0)
            if self.inference_thread.is_alive():
                logging.warning('[Pano2stereo] Inference thread did not finish gracefully')
        
        # Stop RTSP streaming
        if hasattr(self, 'stream_process'):
            self.stop_streaming()
        
        # Clean up FlashDepth processor
        if hasattr(self, 'flashdepth_processor'):
            if hasattr(self.flashdepth_processor, 'cleanup'):
                try:
                    self.flashdepth_processor.cleanup()
                    logging.info('[Pano2stereo] FlashDepth processor cleaned up')
                except Exception as e:
                    logging.error(f'[Pano2stereo] Error cleaning up FlashDepth processor: {e}')
        
        # Clean up GPU memory
        try:
            # Clean up CuPy memory
            if 'cp' in globals():
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            
            # Clean up PyTorch GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logging.info('[Pano2stereo] GPU memory cleaned up')
        except Exception as e:
            logging.error(f'[Pano2stereo] Error cleaning up GPU memory: {e}')
        
        # Clean up precomputed coordinates
        if hasattr(self, 'sphere_coords'):
            del self.sphere_coords
        
        logging.info('[Pano2stereo] Resource cleanup completed')

# =============== Optimized Utility Functions ===============

def load_and_resize_image(img_path):
    """Load and optionally resize image to improve speed"""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"File Not Found: {img_path}")
    
    bgr_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr_array is None:
        raise FileNotFoundError(f"Cannot Read file: {img_path}")
    
    rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
    
    # Optional: Resize image to improve processing speed
    original_height, original_width = rgb_array.shape[:2]
    if (PARAMS.get("TARGET_HEIGHT") and PARAMS.get("TARGET_WIDTH") and 
        (original_height > PARAMS["TARGET_HEIGHT"] or original_width > PARAMS["TARGET_WIDTH"])):
        
        rgb_array = cv2.resize(rgb_array, (PARAMS["TARGET_WIDTH"], PARAMS["TARGET_HEIGHT"]))
        logging.info(f"Image resized from {original_width}x{original_height} to {PARAMS['TARGET_WIDTH']}x{PARAMS['TARGET_HEIGHT']}")
    
    logging.info(f"Input Image: {img_path}")
    logging.info(f"Image Size: {rgb_array.shape[1]}x{rgb_array.shape[0]}")
    return rgb_array

def save_results_optimized(left, right, left_repaired, right_repaired, depth, output_dir):
    """Optimized result saving"""
    # Save multiple images in parallel
    def save_image(img, path, is_gpu=True):
        if is_gpu and hasattr(img, 'get'):
            img = img.get()
        elif is_gpu:
            img = cp.asnumpy(img)
        
        if len(img.shape) == 3 and img.shape[2] == 3:  # RGB image
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), img_bgr)
        else:
            cv2.imwrite(str(path), img)
    
    output_path = Path(output_dir)
    
    # Save main images
    save_image(left, output_path / "left.png")
    save_image(right, output_path / "right.png")
    save_image(left_repaired, output_path / "left_repaired.png", is_gpu=False)
    save_image(right_repaired, output_path / "right_repaired.png", is_gpu=False)
    
    # Fix depth map saving, eliminate warnings
    try:
        # Correctly get depth data
        if hasattr(depth, 'get'):
            depth_cpu = depth.get()
        elif isinstance(depth, cp.ndarray):
            depth_cpu = cp.asnumpy(depth)
        else:
            depth_cpu = depth
        
        # Ensure numpy array
        if not isinstance(depth_cpu, np.ndarray):
            depth_cpu = np.array(depth_cpu)
        
        # Correctly handle depth map data type and range
        if depth_cpu.dtype != np.uint8:
            # Get valid depth values (exclude inf and nan)
            valid_mask = np.isfinite(depth_cpu)
            if np.any(valid_mask):
                depth_min = np.min(depth_cpu[valid_mask])
                depth_max = np.max(depth_cpu[valid_mask])
                
                # Normalize to 0-255 range
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
        
        # Save depth map (now correct uint8 format)
        cv2.imwrite(str(output_path / 'depth.png'), depth_normalized)
        
        if 'valid_mask' in locals() and np.any(valid_mask):
            logging.info(f"Depth map saved (range: {depth_min:.3f}-{depth_max:.3f}m → 0-255)")
        else:
            logging.info("Depth map saved (uint8 format)")
            
    except Exception as e:
        logging.warning(f"Depth map save failed: {e}")


def save_depth_visualization(depth, stereo_dir, frame_count, generator, critical_depth):
    """Save a gray depth image with an annotated colorbar appended on the right.

    - depth: NumPy or CuPy array (H,W) or (H,W,1)
    - stereo_dir: Path-like directory where files will be saved
    - frame_count: integer for filename formatting
    - generator: Pano2stereo instance (used to read generator.critical_depth)
    - critical_depth: fallback critical depth value
    """
    try:
        # Convert to numpy
        if 'cp' in globals() and isinstance(depth, cp.ndarray):
            depth_np = cp.asnumpy(depth)
        else:
            depth_np = np.asarray(depth)

        # Squeeze channel if necessary
        if depth_np.ndim == 3:
            if depth_np.shape[2] == 1:
                depth_np = depth_np[..., 0]
            else:
                depth_np = depth_np[..., 0]

        # Determine critical depth
        try:
            critical = getattr(generator, 'critical_depth', None)
        except Exception:
            critical = None
        if critical is None:
            critical = critical_depth

        # Clip and normalize to [0, critical]
        depth_clipped = np.where(np.isfinite(depth_np), depth_np, 0.0)
        depth_clipped = np.clip(depth_clipped, 0.0, float(critical))
        if float(critical) > 0.0:
            norm = depth_clipped.astype(np.float32) / float(critical)
        else:
            norm = np.zeros_like(depth_clipped, dtype=np.float32)
        norm = np.clip(norm, 0.0, 1.0)

        gray = (norm * 255.0).astype(np.uint8)

        # Generate GRAYSCALE colorbar (match gray image) with a right-side label area
        h, w = gray.shape[:2]
        cb_width_base = max(40, w // 12)
        label_area = 70  # space to the right of the bar for labels
        cb_width = cb_width_base + label_area

        # Vertical gradient: top=near(0), bottom=far(critical)
        gradient = np.linspace(1.0, 0.0, h, dtype=np.float32)
        gradient_gray = (gradient * 255.0).astype(np.uint8)
        # Create a grayscale bar (H x cb_width_base)
        bar = np.tile(gradient_gray[:, None], (1, cb_width_base))
        bar_rgb = np.stack([bar, bar, bar], axis=-1)

        # Create full colorbar area with label background (white)
        cb_full = np.ones((h, cb_width, 3), dtype=np.uint8) * 255
        cb_full[:, :cb_width_base, :] = bar_rgb

        # Annotate ticks and labels on the label area (white background) for readability
        num_ticks = 5
        for i in range(num_ticks):
            y = int(i * (h - 1) / (num_ticks - 1))
            # Tick on the bar portion (black)
            cv2.line(cb_full, (0, y), (min(12, cb_width_base - 1), y), (0, 0, 0), 1)
            val = (1.0 - (i / (num_ticks - 1))) * float(critical)
            # Label on the right label area
            label = f"{val:.2f}m"
            txt_x = cb_width_base + 6
            txt_y = y + 4
            # Draw white outline then black text for good contrast
            cv2.putText(cb_full, label, (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 3, cv2.LINE_AA)
            cv2.putText(cb_full, label, (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)

        cb_rgb = cb_full  # already RGB-like (white labels area)

        # Assemble output image: gray_rgb (left) + cb_rgb (right)
        gray_rgb = np.stack([gray, gray, gray], axis=-1)
        try:
            combined = np.concatenate([gray_rgb, cb_rgb], axis=1)
        except Exception:
            combined = gray_rgb

        depth_gray_path = Path(stereo_dir) / f"depth_gray_{frame_count:03d}.png"

        # Sanity: log mapping between depth values and bar endpoints to detect inversion
        try:
            # combined: left gray_rgb, right cb_rgb
            top_bar_pixel = combined[0, gray_rgb.shape[1] // 2, 0] if combined.shape[1] > gray_rgb.shape[1] else combined[0, 0, 0]
            bottom_bar_pixel = combined[-1, gray_rgb.shape[1] // 2, 0] if combined.shape[1] > gray_rgb.shape[1] else combined[-1, 0, 0]
            logging.info(f"Depth->gray mapping: 0.0 -> black (0), {critical} -> white (255). Colorbar endpoints (top,bottom) pixel values: ({int(top_bar_pixel)},{int(bottom_bar_pixel)})")
        except Exception:
            pass

        cv2.imwrite(str(depth_gray_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        logging.info(f"Saved depth visualization (gray+colorbar): {depth_gray_path.name}")
    except Exception as e:
        logging.warning(f"Failed to save depth visualization: {e}")


def estimate_depth_optimized(generator, rgb_array):
    """Optimized depth estimation (using FlashDepth)"""
    # Convert RGB image to torch tensor
    if isinstance(rgb_array, np.ndarray):
        rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    else:
        rgb_tensor = rgb_array
    rgb_tensor = rgb_tensor.to(generator.device)
    
    # Use FlashDepthProcessor for inference
    # Assume FlashDepthProcessor has process_single_image method
    depth = generator.flashdepth_processor.process_single_image(rgb_tensor)
    
    # Update self.depth
    generator.flashdepth_processor.depth = depth
    
    return cp.asarray(depth.cpu().numpy(), dtype=cp.float32)

def post_process_optimized(output_dir, width, height):
    """Optimized post-processing (optional red-cyan 3D function)"""
    if not PARAMS["ENABLE_POST_PROCESS"]:
        logging.info("Post-processing disabled for speed")
        return
        
    output_dir = Path(output_dir).absolute()
    try:
        # Optional: Generate red-cyan 3D image
        if PARAMS["ENABLE_RED_CYAN"]:
            # Prefer using the stereoscopy Python API (faster, no subprocess)
            try:
                left_path = output_dir / "left_repaired.png"
                right_path = output_dir / "right_repaired.png"

                left_img = Image.open(str(left_path)).convert('RGB')
                right_img = Image.open(str(right_path)).convert('RGB')

                # create_anaglyph expects a list of two PIL images
                anaglyph = stereoscopy.create_anaglyph([left_img, right_img], method="color", color_scheme="red-cyan", luma_coding="rgb")
                anaglyph.save(str(output_dir / "red_cyan.png"))
                logging.info("Red-cyan 3D image generated (via stereoscopy package)")
            except Exception as e:
                logging.warning(f"stereoscopy package failed or not available: {e}; falling back to StereoscoPy subprocess")
                try:
                    subprocess.run([
                        "StereoscoPy",
                        "-S", "5", "0",
                        "-a",
                        "-m", "color",
                        "--cs", "red-cyan",
                        "--lc", "rgb",
                        str(output_dir / "left_repaired.png"),
                        str(output_dir / "right_repaired.png"),
                        str(output_dir / "red_cyan.png")
                    ], check=True, capture_output=True)
                    logging.info("Red-cyan 3D image generated (via subprocess StereoscoPy)")
                except subprocess.CalledProcessError as e:
                    logging.error(f"Red-cyan generation failed (subprocess): {e.returncode}")
        else:
            logging.info("Red-cyan 3D generation skipped (disabled for speed)")

        # Generate left-right stitched stereo image
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",  # Reduce log output
            "-i", str(output_dir / "left_repaired.png"),
            "-i", str(output_dir / "right_repaired.png"),
            "-filter_complex", f"[0:v][1:v]hstack", 
            "-frames:v", "1", 
            str(output_dir / "stereo.jpg")
        ], check=True, capture_output=True)

        # Optional: Generate video
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

# =============== Main Function ===============
def main():
    """Optimized main function (focused on core timing)"""
    profiler.reset()
    # print(PARAMS)
    
    # Configuration parameters (not participating in core timing)
    IPD = PARAMS["IPD"]
    PRO_RADIUS = PARAMS["PRO_RADIUS"]
    PIXEL = PARAMS["PIXEL"]
    CRITICAL_DEPTH = min(PARAMS["CRITICAL_DEPTH"], cal_critical_depth(IPD, PIXEL))
    URL = PARAMS.get("URL")  # Use get() to safely handle missing URL
    
    try:
        # Determine input source and get dimensions
        if URL:
            logging.info(f"Using video stream: {URL}")  
        else:
            raise ValueError("NO URL provided - please set URL in configs/pano.yaml")

        # 1. Generator initialization (including model warm-up, not participating in core timing)
        logging.info(f'[Parameters] IPD: {IPD}m, Critical Depth: {CRITICAL_DEPTH:.2f}m')
        # Force log flush
        for handler in logging.getLogger().handlers:
            handler.flush()
            
        generator = Pano2stereo(
            IPD=IPD, 
            pro_radius=PRO_RADIUS, 
            pixel=PIXEL, 
            critical_depth=CRITICAL_DEPTH, 
            url=URL
        )
        
        # Create stereo_img directory and initialize frame counter
        stereo_dir = Path(generator.outputdir) / "stereo_img"
        stereo_dir.mkdir(parents=True, exist_ok=True)
        frame_count = 0
        
        # ================== 2. Main Loop ==================
        while generator.running:
            # If FlashDepthProcessor has finished (e.g., reached max_frames), stop the main generator loop
            try:
                if generator.flashdepth_processor.stopped:
                    logging.info('[Pano2stereo] Detected FlashDepthProcessor stopped flag; Stopping')
                    generator.running = False
                    break
            except Exception:
                # Defensive: ignore attribute errors and continue
                pass

            # 2.1 Generate stereo pairs from depth and RGB                 
            # self.pred = [depth_pred, original_frame]  # Store latest depth map & frame（Tensor, shape [H, W, C] in BGR)
            depth, bgr = torch_to_cupy(generator.flashdepth_processor.pred[0], generator.flashdepth_processor.pred[1])

            # TODO: the physical meaning of depth = 100/x (m)?? 
            depth = generator.disparity_to_depth(disparity_map=depth, critical_depth=CRITICAL_DEPTH)  # Convert to meters and clamp
            
            # Compute and log basic statistics for the depth map we just received.
            try:
                # depth may be a CuPy array or NumPy array; handle both
                if 'cp' in globals() and isinstance(depth, cp.ndarray):
                    # Move minimal statistics to CPU without copying entire array if possible
                    # Use CuPy's reduction which is efficient on GPU
                    # Filter out non-finite values
                    finite_mask = cp.isfinite(depth)
                    if cp.any(finite_mask):
                        depth_min = float(cp.min(depth[finite_mask]).get())
                        depth_max = float(cp.max(depth[finite_mask]).get())
                        depth_mean = float(cp.mean(depth[finite_mask]).get())
                    else:
                        depth_min = float('nan')
                        depth_max = float('nan')
                        depth_mean = float('nan')
                else:
                    # Assume NumPy array-like
                    import numpy as _np
                    depth_np = _np.asarray(depth)
                    finite_mask = _np.isfinite(depth_np)
                    if _np.any(finite_mask):
                        depth_min = float(_np.min(depth_np[finite_mask]))
                        depth_max = float(_np.max(depth_np[finite_mask]))
                        depth_mean = float(_np.mean(depth_np[finite_mask]))
                    else:
                        depth_min = float('nan')
                        depth_max = float('nan')
                        depth_mean = float('nan')
                logging.info(f"Depth stats — min: {depth_min:.6g}, max: {depth_max:.6g}, mean: {depth_mean:.6g}")
            except Exception as _e:
                logging.warning(f"Failed to compute depth statistics: {_e}")

            # Save depth visualization (gray + annotated colorbar)
            save_depth_visualization(depth, stereo_dir, frame_count, generator, CRITICAL_DEPTH)
            
            rgb = bgr[..., ::-1]  # Convert BGR to RGB
            logging.info(f"Depth shape: {depth.shape}, RGB shape: {rgb.shape}")
            # At this point we trust FlashDepth: rgb and depth are same resolution
            left, right = generator.generate_stereo_pair(rgb_data=rgb, depth_data=depth)
        
            # 2.2 🔧 Image repair (core timing - parallel processing)
            # Parallel repair of left and right images            
            # Use thread pool for parallel processing of left and right images
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit left and right image repair tasks
                future_left = executor.submit(generator.repair_black_regions, left)
                future_right = executor.submit(generator.repair_black_regions, right)
                
                # Get results
                left_repaired = future_left.result()
                right_repaired = future_right.result()

            # Save and post-process (not participating in core timing)
            if generator.save_result:
                output_dir = generator.outputdir
                save_results_optimized(left, right, left_repaired, right_repaired, depth, output_dir)
                post_process_optimized(output_dir, generator.width, generator.height)

            # 2.3  Streaming
            # Stream to target URL
            stereo_image = np.hstack((cp.asnumpy(left_repaired), cp.asnumpy(right_repaired)))
            logging.info(f"Streaming frame size: {stereo_image.shape[1]}x{stereo_image.shape[0]}")
            
            # 2.4 Saving results

            # Generate Red-Cyan anaglyph
            # Convert left_repaired/right_repaired to CPU numpy arrays
            left_np = cp.asnumpy(left_repaired)
            right_np = cp.asnumpy(right_repaired)

            # Convert to PIL Images and call stereoscopy API
            left_img = Image.fromarray(left_np)
            right_img = Image.fromarray(right_np)

            anaglyph = stereoscopy.create_anaglyph([left_img, right_img], method="color", color_scheme="red-cyan", luma_coding="rgb")
            red_cyan_path = stereo_dir / f"red_cyan_{frame_count:03d}.png"
            anaglyph.save(str(red_cyan_path))

            # red_cyan = np.zeros_like(left_repaired, dtype=np.uint8)
            # red_cyan[:, :, 0] = left_repaired[:, :, 0]  # Red channel from left eye
            # red_cyan[:, :, 1] = right_repaired[:, :, 1]  # Green channel from right eye
            # red_cyan[:, :, 2] = right_repaired[:, :, 2]  # Blue channel from right eye
            
            # Save Red-Cyan anaglyph
            # red_cyan_path = stereo_dir / f"red_cyan_{frame_count:03d}.png"
            # cv2.imwrite(str(red_cyan_path), cv2.cvtColor(red_cyan, cv2.COLOR_RGB2BGR))
            
            # Save stereo_image to local before streaming
            stereo_path = stereo_dir / f"stereo_{frame_count:03d}.png"
            cv2.imwrite(str(stereo_path), cv2.cvtColor(stereo_image, cv2.COLOR_RGB2BGR))

            frame_count += 1
            # Stream the stereo image via RTSP
            generator.stream_frame(stereo_image)
            # generator.stream_frame(red_cyan_path)  # Stream the red-cyan image 

        # Memory cleanup
        del rgb, depth, left, right
        cp.get_default_memory_pool().free_all_blocks()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logging.info("=== Processing completed successfully ===")
        # Final flush
        for handler in logging.getLogger().handlers:
            handler.flush()
        
        # Stop streaming
        generator.stop_streaming()
            
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        # Stop streaming on error
        if 'generator' in locals():
            generator.stop_streaming()
        raise

if __name__ == "__main__":
    logging_setup()
    main()
