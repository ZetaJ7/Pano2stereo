import torch
import numpy as np
import threading
import time
import math
import os
import cv2
import sys
import logging
import subprocess
from pathlib import Path
import queue
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
                    return cp.from_dlpack(_dlpack.to_dlpack(x))
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
        # Number of lines last rendered to terminal by render_live
        self._last_live_lines = 0
    
    def reset(self):
        """Reset timer data"""
        self.timings.clear()
        self.active_timers.clear()
    
    @contextmanager
    def timer(self, name: str):
        """Context manager for timing a code block.

        Usage:
            with profiler.timer('task'):
                do_work()
        """
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            lst = self.timings.setdefault(name, [])
            lst.append(elapsed)
            # Compute simple statistics
            count = len(lst)
            avg = sum(lst) / count if count > 0 else elapsed
            try:
                # Use INFO to ensure visibility in logs; include count and average
                logging.info(f"[Timer] {name}: {elapsed:.6f}s (count={count}, avg={avg:.6f}s)")
            except Exception:
                # Best-effort: ignore logging errors
                pass

    def render_live(self, max_lines: int = 10):
        """Render a compact profiler summary to the terminal, updating in-place.

        This uses ANSI cursor movement to overwrite the previous block so the
        profiler output appears to update instead of appending new lines.
        """
        try:
            # Prepare sorted stats: by average descending
            items = []
            for name, lst in self.timings.items():
                if not lst:
                    continue
                last = lst[-1]
                count = len(lst)
                avg = sum(lst) / count
                items.append((name, last, count, avg))

            if not items:
                return

            items.sort(key=lambda x: x[3], reverse=True)  # sort by avg desc
            lines = []
            for name, last, count, avg in items[:max_lines]:
                lines.append(f"[Timer] {name}: {last:.6f}s (count={count}, avg={avg:.6f}s)")

            out = sys.stdout

            # Move cursor up to the start of previous block if any
            if self._last_live_lines > 0:
                out.write(f"\x1b[{self._last_live_lines}A")

            # Write new block, clearing each line before writing
            for line in lines:
                out.write("\x1b[2K")  # clear entire line
                out.write(line + "\n")

            # If we previously printed more lines than now, clear the leftover lines
            if self._last_live_lines > len(lines):
                for _ in range(self._last_live_lines - len(lines)):
                    out.write("\x1b[2K\n")

            out.flush()
            self._last_live_lines = len(lines)
        except Exception:
            # Silent fallback if terminal control fails
            pass

# Global profiler instance used throughout the module
profiler = PerformanceProfiler()
    
class Pano2stereo:
    def __init__(self, IPD, pro_radius, pixel, critical_depth, url, red_cyan=True, max_frames=None):
        # Store configuration
        self.IPD = IPD
        self.pro_radius = pro_radius
        self.pixel = pixel
        self.critical_depth = critical_depth
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.stream_redcyan = red_cyan

        # Prepare output directory and urls
        self.outputdir = self.make_output_dir()
        self.url = url
        self.target_url = 'rtsp://10.20.35.30:28552/result'
        self.running = True
        self.save_result = False

        # Initialize RTSP streaming placeholders
        self.stream_process = None
        self.stream_fps = 30  # Default FPS for streaming
        # Background streaming queue and writer thread
        self._stream_queue = queue.Queue(maxsize=8)
        self._stream_writer_thread = None
        self._stream_thread_stop = threading.Event()
        # How many frames between flushes in writer thread
        self._stream_flush_interval = 4
        self._stream_lock = threading.Lock()

        # Load model (FlashDepth)
        logging.info("Initializing FlashDepth model...")
        self.flashdepth_processor = FlashDepthProcessor(config_path="configs/flashdepth.yaml", url=self.url, stream_mode=True, save_depth_png=False, save_frame=False, max_frames=max_frames, run_dir=self.outputdir)

        # Start FlashDepth inference in a separate thread
        self.inference_thread = threading.Thread(target=self.flashdepth_processor.run_inference, daemon=True)
        self.inference_thread.start()
        logging.info('[FlashDepth] Inference thread started in Pano2stereo.__init__')

        # Wait for FlashDepth to produce an initial prediction and read resolution
        logging.info("[Pano2stereo] Waiting for FlashDepth inference to start...")
        while self.flashdepth_processor.pred is None:
            time.sleep(0.02)

        # Read inferred resolution
        self.height = self.flashdepth_processor.pred[0].shape[0]
        self.width = self.flashdepth_processor.pred[0].shape[1]
        logging.info(f"[Pano2stereo] Got Input resolution from FlashDepth: {self.width}x{self.height}")

        # Initialize streaming pipeline now that dimensions are known
        self._init_streaming()

        # Precompute sphere coordinates for forward mapping
        logging.info("[Pano2stereo] Precomputing sphere coordinates...")
        self.sphere_coords = self._precompute_sphere_coords(self.width, self.height)  # (r, theta, phi) for [H,W] resolution, shape [H, W, 3]

        logging.info('[Pano2stereo] Initialization completed')



    def _init_streaming(self):
        """Initialize RTSP streaming pipeline"""
        try:
            # Calculate output resolution
            if self.stream_redcyan:
                output_width = self.width
                output_height = self.height
                logging.info('[Streaming Init] Streaming Red-cyan (single view)')
            else:
                output_width = self.width * 2
                output_height = self.height
                logging.info('[Streaming Init] Streaming Left-Right Pano (side-by-side)')

            # Record the stream resolution for later use in stream_frame
            self.stream_width = int(output_width)
            self.stream_height = int(output_height)
            
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
                # Force output to a widely-supported pixel format/profile to maximize
                # compatibility with clients (ffplay, hardware decoders).
                '-pix_fmt', 'yuv420p',
                '-profile:v', 'baseline',
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
            
            logging.info(f"[Streaming Init] Initializing RTSP stream to {self.target_url}")
            logging.info(f"[Streaming Init] Stream resolution: {output_width}x{output_height} @ {self.stream_fps}fps")

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
            # Start background writer thread if not already running
            try:
                if self._stream_writer_thread is None or not self._stream_writer_thread.is_alive():
                    self._stream_thread_stop.clear()
                    self._stream_writer_thread = threading.Thread(target=self._stream_writer, name='stream-writer', daemon=True)
                    self._stream_writer_thread.start()
            except Exception as e:
                logging.warning(f"Failed to start stream writer thread: {e}")
            
        except Exception as e:
            logging.error(f"Failed to initialize RTSP streaming: {e}")
            self.stream_process = None

    def stream_frame(self, frame):
        """Stream a frame to RTSP"""
        # Prepare frame bytes and enqueue for background writer.
        # Assumes caller guarantees correct size (user requested removal of resize logic).
        try:
            if isinstance(frame, cp.ndarray):
                frame = cp.asnumpy(frame)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            frame = np.ascontiguousarray(frame)
            data = frame.tobytes()

            # Non-blocking enqueue: if queue full, drop the frame (real-time behavior)
            try:
                self._stream_queue.put_nowait(data)
            except queue.Full:
                # Drop frame; count or log at debug level to avoid noisy logs
                logging.debug("Stream queue full, dropping frame")
        except Exception as e:
            logging.error(f"Failed to prepare frame for streaming: {e}")

    def _restart_streaming(self):
        """Restart RTSP streaming"""
        # When restarting, clear any queued frames to avoid stale data
        try:
            while not self._stream_queue.empty():
                try:
                    self._stream_queue.get_nowait()
                except Exception:
                    break
        except Exception:
            pass
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

    def _stream_writer(self):
        """Background writer thread that consumes frames from queue and writes to ffmpeg stdin.

        This avoids blocking the main loop on IO. It flushes every _stream_flush_interval frames.
        """
        frames_since_flush = 0
        while not self._stream_thread_stop.is_set():
            try:
                data = self._stream_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                # Use a lock around process access to avoid races with restart/stop
                with self._stream_lock:
                    if not self.stream_process or self.stream_process.poll() is not None:
                        # Stream is dead; try restart and drop this frame
                        logging.debug("Stream writer detected dead process; restarting")
                        try:
                            self._restart_streaming()
                        except Exception:
                            pass
                        continue
                    try:
                        self.stream_process.stdin.write(data)
                    except BrokenPipeError:
                        logging.warning("ffmpeg stdin broken in writer thread; restarting")
                        try:
                            self._restart_streaming()
                        except Exception:
                            pass
                        continue
                frames_since_flush += 1
                if frames_since_flush >= getattr(self, '_stream_flush_interval', 4):
                    try:
                        with self._stream_lock:
                            if self.stream_process and self.stream_process.stdin:
                                self.stream_process.stdin.flush()
                    except Exception:
                        pass
                    frames_since_flush = 0
            except Exception as e:
                logging.debug(f"Stream writer thread error: {e}")
                # Avoid tight-looping on error
                time.sleep(0.05)

    def stop_streaming(self):
        """Stop RTSP streaming"""
        # Signal writer thread to stop first
        try:
            self._stream_thread_stop.set()
        except Exception:
            pass
        # Join writer thread
        try:
            if self._stream_writer_thread is not None and self._stream_writer_thread.is_alive():
                self._stream_writer_thread.join(timeout=2.0)
        except Exception:
            pass

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
        with profiler.timer("[generate_stereo_pair] Function total"):           
            with profiler.timer("---------------------\n[generate_stereo_pair] Depth fix on circular projection"):
                sphere_l, sphere_r = self.sphere2pano_vectorized(depth_data)
            with profiler.timer("[generate_stereo_pair] Revert sphere to image pixel"):
                coords_l = self.sphere_to_image_coords(sphere_l, self.width, self.height)
                coords_r = self.sphere_to_image_coords(sphere_r, self.width, self.height)
            with profiler.timer("[generate_stereo_pair] Image painting (Z-Buffer)"):
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

        # Also record coordinates (row=y, col=x) of unmapped pixels as CuPy arrays
        # nonzero returns (ys, xs)
        ys_l, xs_l = cp.nonzero(mask_left_unmapped)
        ys_r, xs_r = cp.nonzero(mask_right_unmapped)
        self._last_unmapped_coords = ((ys_l, xs_l), (ys_r, xs_r))

        return left, right

    def repair_black_regions(self, image, side='both'):
        """Repair unmapped (black) regions using recorded self._last_unmapped_coords.

        Strategy:
        - If `self._last_unmapped_coords` exists use a GPU-local fill: for each unmapped pixel,
          replace its RGB with the mean of valid neighbouring pixels within a small window (e.g. 3x3).
        - If GPU path fails or CuPy is unavailable, fall back to the previous CPU-based OpenCV inpaint.

        Arguments:
            image: array-like RGB image (can be CuPy or NumPy)
            side: 'left', 'right' or 'both' to indicate which image's unmapped coords to use.

        Returns:
            Repaired image as NumPy array (uint8, RGB)
        """
        if not PARAMS.get("ENABLE_REPAIR", True):
            # Return NumPy array for consistency
            return cp.asnumpy(image) if 'cp' in globals() and isinstance(image, cp.ndarray) else np.asarray(image)

        # Prefer GPU path: use distance-transform nearest-neighbor remap when cupyx supports it
        try:
            if 'cp' not in globals() or not isinstance(image, cp.ndarray):
                image_gpu = cp.asarray(image, dtype=cp.uint8)
            else:
                image_gpu = image.astype(cp.uint8)

            H, W = image_gpu.shape[0], image_gpu.shape[1]

            # Determine which sides to repair: build fill mask from recorded coords
            coords = None
            if hasattr(self, '_last_unmapped_coords') and self._last_unmapped_coords is not None:
                if side == 'left':
                    coords = self._last_unmapped_coords[0]
                elif side == 'right':
                    coords = self._last_unmapped_coords[1]
                else:
                    ys_l, xs_l = self._last_unmapped_coords[0]
                    ys_r, xs_r = self._last_unmapped_coords[1]
                    ys = cp.concatenate([ys_l, ys_r])
                    xs = cp.concatenate([xs_l, xs_r])
                    coords = (ys, xs)

            if coords is None or len(coords) != 2:
                raise RuntimeError("No unmapped coords available for repair; falling back to CPU inpaint")

            ys, xs = coords

            # Build fill mask (True where we need to fill)
            fill_mask = cp.zeros((H, W), dtype=cp.bool_)
            if ys.size > 0:
                fill_mask[ys, xs] = True
            if not fill_mask.any():
                # nothing to fill
                return cp.asnumpy(image_gpu.astype(cp.uint8))

            # Valid pixels are those with any non-zero channel
            valid_mask = cp.any(image_gpu != 0, axis=-1)

            # If there are no valid pixels, fallback
            if not valid_mask.any():
                raise RuntimeError("No valid pixels available to sample from; falling back to CPU inpaint")

            # Use cupyx distance transform to compute nearest valid pixel indices
            try:
                from cupyx.scipy.ndimage import distance_transform_edt
                # distance_transform_edt expects background==0; compute for valid_mask==False
                # We request indices of nearest non-zero (i.e. valid) locations
                dist, indices = distance_transform_edt(~valid_mask, return_indices=True)
                # indices shape is (ndim, H, W)
                idx_y = indices[0].astype(cp.int32)
                idx_x = indices[1].astype(cp.int32)

                # Remap entire image from nearest valid indices
                remapped = image_gpu[idx_y, idx_x]

                # Only replace fill_mask pixels
                out = image_gpu.copy()
                out[fill_mask] = remapped[fill_mask]

                return cp.asnumpy(out.astype(cp.uint8))
            except Exception as e_dist:
                # If cupyx distance transform not available or fails, raise to trigger CPU fallback
                raise RuntimeError(f"cupyx distance_transform_edt failed: {e_dist}")

        except Exception as e:
            logging.warning(f"GPU repair path failed or unavailable ({e}), falling back to CPU inpaint")

            # CPU fallback: use previous inpaint approach on full image
            image_cpu = cp.asnumpy(image).astype(np.uint8) if 'cp' in globals() and isinstance(image, cp.ndarray) else np.asarray(image).astype(np.uint8)
            img_bgr = cv2.cvtColor(image_cpu, cv2.COLOR_RGB2BGR)

            # If _last_unmapped_coords exist, construct mask from them; otherwise detect black pixels
            mask = None
            try:
                if hasattr(self, '_last_unmapped_coords') and self._last_unmapped_coords is not None:
                    ys_l, xs_l = cp.asnumpy(self._last_unmapped_coords[0])
                    ys_r, xs_r = cp.asnumpy(self._last_unmapped_coords[1])
                    ys = np.concatenate([ys_l, ys_r]).astype(np.int32)
                    xs = np.concatenate([xs_l, xs_r]).astype(np.int32)
                    mask = np.zeros((image_cpu.shape[0], image_cpu.shape[1]), dtype=np.uint8)
                    mask[ys, xs] = 255
                else:
                    mask = np.uint8(cv2.cvtColor(image_cpu, cv2.COLOR_RGB2GRAY) == 0) * 255
            except Exception:
                mask = np.uint8(cv2.cvtColor(image_cpu, cv2.COLOR_RGB2GRAY) == 0) * 255

            if np.sum(mask) == 0:
                # Nothing to repair
                return image_cpu

            repaired = cv2.inpaint(img_bgr, mask, inpaintRadius=2, flags=cv2.INPAINT_NS)
            return cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)

    def repair_black_regions_pair(self, left, right):
        """Repair left and right images in a single combined call.

        This concatenates left and right horizontally (keeping on GPU when possible),
        runs the GPU-preferred repair once (distance transform remap), then splits
        and returns the repaired left and right images. This avoids duplicated
        distance-transform / remap work when both views share the same valid pixels.
        """
        # Convert inputs to CuPy if available to prefer GPU path
        left_gpu = cp.asarray(left) if not isinstance(left, cp.ndarray) else left
        right_gpu = cp.asarray(right) if not isinstance(right, cp.ndarray) else right

        # Concatenate horizontally (W_left + W_right)
        try:
            combined = cp.concatenate([left_gpu, right_gpu], axis=1)
            combined_repaired = None

            # Try to reuse existing repair_black_regions code path by passing side='both'
            # but operate directly on the concatenated image. We temporarily set
            # _last_unmapped_coords to combined coords so the method will build a fill mask correctly.
            # Build combined unmapped coords from existing per-eye coords if present
            if hasattr(self, '_last_unmapped_coords') and self._last_unmapped_coords is not None:
                (ys_l, xs_l), (ys_r, xs_r) = self._last_unmapped_coords
                # shift right xs by left width
                shift = left_gpu.shape[1]
                xs_r_shifted = xs_r + int(shift)
                ys = cp.concatenate([ys_l, ys_r])
                xs = cp.concatenate([xs_l, xs_r_shifted])
                # Temporarily override
                orig_coords = self._last_unmapped_coords
                self._last_unmapped_coords = ( (ys, xs), (cp.array([], dtype=cp.int32), cp.array([], dtype=cp.int32)) )
            else:
                orig_coords = None

            try:
                combined_repaired = self.repair_black_regions(combined, side='both')
            finally:
                # restore original coords
                if orig_coords is not None:
                    self._last_unmapped_coords = orig_coords

            # combined_repaired returned as numpy uint8; convert back to cupy if needed then split
            if isinstance(combined_repaired, np.ndarray):
                combined_np = combined_repaired
            else:
                combined_np = cp.asnumpy(combined_repaired)

            w_left = left.shape[1] if not isinstance(left, cp.ndarray) else left_gpu.shape[1]
            left_out = combined_np[:, :w_left, :]
            right_out = combined_np[:, w_left:, :]

            # return as CuPy arrays to keep consistency with pipeline (main expects cp arrays)
            return cp.asarray(left_out), cp.asarray(right_out), combined_np

        except Exception:
            # Fallback: run repair separately (safer but slower)
            left_r = self.repair_black_regions(left, side='left')
            right_r = self.repair_black_regions(right, side='right')
            return cp.asarray(left_r), cp.asarray(right_r)

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

def logging_setup(log_file: str = 'logs/pano2stereo.log'):
    """Ensure logging is configured (idempotent). Call at program start."""
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    except Exception:
        pass
    # Reconfigure root logger handlers if not already configured
    root = logging.getLogger()
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        root.setLevel(logging.INFO)
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'))
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        root.addHandler(fh)
        root.addHandler(sh)


def save_results_visualization(left, right, left_repaired, right_repaired, depth, output_dir, pano_repair=None, frame_idx=None, generator=None):
    """Optimized result saving (extended signature).

    This function preserves previous behavior but also saves the stitched
    `pano_repair` if provided, and can call stereoscopy.create_anaglyph when
    requested via PARAMS["ENABLE_RED_CYAN"].
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def _to_numpy(x):
        if 'cp' in globals() and isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
        return np.asarray(x)

    try:
        left_np = _to_numpy(left)
        right_np = _to_numpy(right)
        left_rep_np = _to_numpy(left_repaired)
        right_rep_np = _to_numpy(right_repaired)

        # Save images
        cv2.imwrite(str(output_path / f"left_{frame_idx:05d}.png" if frame_idx is not None else output_path / "left.png"), cv2.cvtColor(left_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_path / f"right_{frame_idx:05d}.png" if frame_idx is not None else output_path / "right.png"), cv2.cvtColor(right_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_path / f"left_repaired_{frame_idx:05d}.png" if frame_idx is not None else output_path / "left_repaired.png"), cv2.cvtColor(left_rep_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_path / f"right_repaired_{frame_idx:05d}.png" if frame_idx is not None else output_path / "right_repaired.png"), cv2.cvtColor(right_rep_np, cv2.COLOR_RGB2BGR))

        # Save stitched pano_repair if available
        if pano_repair is not None:
            pano_np = _to_numpy(pano_repair)
            cv2.imwrite(str(output_path / f"stereo_{frame_idx:05d}.png" if frame_idx is not None else output_path / "stereo.png"), cv2.cvtColor(pano_np, cv2.COLOR_RGB2BGR))

        # Optional red-cyan generation (GPU-first using gpu_create_anaglyph)
        if PARAMS.get("ENABLE_RED_CYAN", False) and generator is not None:
            try:
                anaglyph_np = gpu_create_anaglyph(left_rep_np, right_rep_np)
                anaglyph_path = output_path / (f"red_cyan_{frame_idx:05d}.png" if frame_idx is not None else "red_cyan.png")
                if anaglyph_np is not None:
                    try:
                        cv2.imwrite(str(anaglyph_path), cv2.cvtColor(anaglyph_np, cv2.COLOR_RGB2BGR))
                        logging.info(f"Saved GPU red-cyan anaglyph: {anaglyph_path.name}")
                    except Exception as e:
                        logging.warning(f"Failed to write GPU anaglyph to disk: {e}")
                else:
                    # Fallback to stereoscopy if GPU routine returned None
                    try:
                        left_img = Image.fromarray(left_rep_np.astype(np.uint8))
                        right_img = Image.fromarray(right_rep_np.astype(np.uint8))
                        anaglyph = stereoscopy.create_anaglyph([left_img, right_img], method="color", color_scheme="red-cyan", luma_coding="rgb")
                        anaglyph.save(str(output_path / f"red_cyan_{frame_idx:05d}.png" if frame_idx is not None else output_path / "red_cyan.png"))
                    except Exception as e:
                        logging.warning(f"Fallback stereoscopy anaglyph generation failed: {e}")
            except Exception as e:
                logging.warning(f"GPU red-cyan generation failed in save_results_visualization: {e}")

        # Save depth visualization using the centralized helper to keep behavior consistent
        try:
            # Use the same stereo_dir/output_path and frame index semantics
            stereo_dir = output_path
            # call the helper which will handle CuPy/NumPy conversion and colorbar
            save_depth_visualization(depth, stereo_dir, frame_idx if frame_idx is not None else 0, generator, getattr(generator, 'critical_depth', PARAMS.get('CRITICAL_DEPTH', 9999)))
        except Exception as e:
            logging.warning(f"Failed to save depth via save_depth_visualization: {e}")

    except Exception as e:
        logging.warning(f"save_results_optimized failed: {e}")


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

        depth_gray_path = Path(stereo_dir) / f"depth_gray_{frame_count:05d}.png"

        # Sanity: log mapping between depth values and bar endpoints to detect inversion
        try:
            # combined: left gray_rgb, right cb_rgb
            top_bar_pixel = combined[0, gray_rgb.shape[1] // 2, 0] if combined.shape[1] > gray_rgb.shape[1] else combined[0, 0, 0]
            bottom_bar_pixel = combined[-1, gray_rgb.shape[1] // 2, 0] if combined.shape[1] > gray_rgb.shape[1] else combined[-1, 0, 0]
            # logging.info(f"Depth->gray mapping: 0.0 -> black (0), {critical} -> white (255). Colorbar endpoints (top,bottom) pixel values: ({int(top_bar_pixel)},{int(bottom_bar_pixel)})")
        except Exception:
            pass

        cv2.imwrite(str(depth_gray_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        # logging.info(f"Saved depth visualization (gray+colorbar): {depth_gray_path.name}")
    except Exception as e:
        logging.warning(f"Failed to save depth visualization: {e}")


def gpu_create_anaglyph(left_img, right_img):
    """Create a red-cyan anaglyph on GPU using CuPy.

    Simple color anaglyph approximation:
      - output R = left.R
      - output G = right.G * 0.0 (we set from right) or mix
      - output B = right.B

    left_img/right_img may be NumPy arrays or CuPy arrays. Returns a NumPy uint8 RGB image.
    """
    try:
        # Ensure on GPU
        if 'cp' in globals() and isinstance(left_img, cp.ndarray):
            l_gpu = left_img.astype(cp.uint8)
        else:
            l_gpu = cp.asarray(left_img, dtype=cp.uint8)

        if 'cp' in globals() and isinstance(right_img, cp.ndarray):
            r_gpu = right_img.astype(cp.uint8)
        else:
            r_gpu = cp.asarray(right_img, dtype=cp.uint8)

        # Ensure shape and channels
        if l_gpu.ndim == 2:
            l_gpu = cp.stack([l_gpu, l_gpu, l_gpu], axis=-1)
        if r_gpu.ndim == 2:
            r_gpu = cp.stack([r_gpu, r_gpu, r_gpu], axis=-1)

        # Compose anaglyph: R from left, G and B from right
        out_gpu = cp.empty_like(l_gpu)
        out_gpu[..., 0] = l_gpu[..., 0]
        # For a slightly better perceptual result, mix right green with left green a bit
        out_gpu[..., 1] = r_gpu[..., 1]
        out_gpu[..., 2] = r_gpu[..., 2]

        # Transfer back to CPU for saving
        out_np = cp.asnumpy(out_gpu)
        return out_np.astype(np.uint8)
    except Exception as e:
        logging.warning(f"GPU anaglyph generation failed: {e}")
        # Fallback to CPU composition
        try:
            l = np.asarray(left_img).astype(np.uint8)
            r = np.asarray(right_img).astype(np.uint8)
            out = np.zeros_like(l)
            out[..., 0] = l[..., 0]
            out[..., 1] = r[..., 1]
            out[..., 2] = r[..., 2]
            return out
        except Exception:
            return None

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
            url=URL,
            # red_cyan = False,
            red_cyan = True,
            max_frames = 600
        )
        
        # Create visualization directory and initialize frame counter
        stereo_dir = Path(generator.outputdir) / "visualization"
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
            
            rgb = bgr[..., ::-1]  # Convert BGR to RGB
            # logging.info(f"[MAIN LOOP] Depth shape: {depth.shape}, RGB shape: {rgb.shape}")
            # At this point we trust FlashDepth: rgb and depth are same resolution
            with profiler.timer("[MAIN LOOP] RGB+Depth to Streaming"):
                with profiler.timer("[Pano2stereo] RGB+Depth to Stereo"):
                    left, right = generator.generate_stereo_pair(rgb_data=rgb, depth_data=depth)
            
                # 2.2 🔧 Image repair (single combined call - GPU-friendly)
                # Combine left and right repair into a single call to avoid duplicate GPU work
                try:
                    with profiler.timer("[Pano2stereo] Repair_black_regions_pair"):
                        left_repaired, right_repaired, pano_repair = generator.repair_black_regions_pair(left, right)
                except Exception as e:
                    logging.warning(f"[MAIN LOOP] Combined repair failed, falling back to parallel repair: {e}")
                    # Fall back to previous behavior (parallel threads) if combined fails
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                        future_left = executor.submit(generator.repair_black_regions, left)
                        future_right = executor.submit(generator.repair_black_regions, right)
                        left_repaired = future_left.result()
                        right_repaired = future_right.result()
                        pano_repair = np.concatenate([left_repaired, right_repaired], axis=1)

                # Save and post-process (not participating in core timing)
                output_dir = Path(generator.outputdir) / (f"visualization/")
                # if (1) and (frame_count % 20 == 0):
                #     save_results_visualization(left, right, left_repaired, right_repaired, depth, output_dir, pano_repair=pano_repair, frame_idx=frame_count, generator=generator)

                # Optional: fast GPU-generated red-cyan anaglyph           
                # 2.3  Streaming
                if generator.stream_redcyan:  
                    with profiler.timer("[MAIN LOOP] Generating Red-Cyan Anaglyph"):
                        anaglyph_np = gpu_create_anaglyph(left_repaired, right_repaired)
                    # if (False) and (frame_count % 20 == 0):
                    #     anaglyph_path = Path(output_dir) / (f"red_cyan_{frame_count:05d}.png")
                    #     cv2.imwrite(str(anaglyph_path), cv2.cvtColor(anaglyph_np, cv2.COLOR_RGB2BGR))
                    #     logging.info(f"[MAIN LOOP] Saved GPU red-cyan anaglyph: {anaglyph_path.name}")
                    # else:
                    #     logging.debug("[MAIN LOOP] GPU anaglyph generation returned None; skipping save")       
                        # Stream red_cyan to target URL
                    # logging.info(f"[MAIN LOOP] Streaming frame size: {anaglyph_np.shape[1]}x{anaglyph_np.shape[0]}")
                    with profiler.timer("[MAIN LOOP] Streaming Red-Cyan Anaglyph"):
                        generator.stream_frame(anaglyph_np)  
                else:
                    # Stream Pano(left+right)to target URL
                    # logging.info(f"[MAIN LOOP]Streaming frame size: {pano_repair.shape[1]}x{pano_repair.shape[0]}")
                    with profiler.timer("[MAIN LOOP] Streaming Left-Right Pano"):
                        generator.stream_frame(pano_repair)  
            
            # 2.4 Frame_count and save
            # save to PNGs
            if True:
                (output_dir / "stream_frames").mkdir(parents=True, exist_ok=True)
                if generator.stream_redcyan:
                    cv2.imwrite(str(output_dir / f"stream_frames/red_cyan_{frame_count:05d}.png"), cv2.cvtColor(anaglyph_np.astype(np.uint8), cv2.COLOR_RGB2BGR))
                else:
                    cv2.imwrite(str(output_dir / f"stream_frames/pano_{frame_count:05d}.png"), cv2.cvtColor(cp.asnumpy(pano_repair).astype(np.uint8), cv2.COLOR_RGB2BGR))
            frame_count += 1
            # Update live profiler output in terminal (overwrites previous block)
            try:
                profiler.render_live(max_lines=12)
            except Exception:
                pass


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
