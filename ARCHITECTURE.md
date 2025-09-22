# Pano2Stereo 架构与实现说明（合并版）


## 1. 整体架构总览

- 单一视频流由 `FrameReader` 解码，完成 BGR→RGB 与 14 倍数对齐的尺寸预处理。
- `FrameReader` 将处理后的帧直接送入 `FlashDepthProcessor` 的外部帧队列（零中转），并维护 `current_frame` 供主循环使用。
- `FlashDepthProcessor.run_inference(external_frame_mode=True)` 从队列取帧，统一走流式推理路径，输出最新深度 `pred`（torch.float32, CUDA）。
- `Pano2stereo._depth_update_worker` 高频读取 `pred`，在读取时完成 DLPack 零拷贝转为 CuPy 并缓存为 `latest_depth`。
- 主循环专注于：取最新 `current_frame` + `latest_depth` → 立体图生成（GPU）→ 修复 → 流式输出。

```
Video Stream ──► FrameReader ──► (send to) FlashDepthProcessor.frame_queue
                   │                                  │
                   └──► current_frame (RGB, resized)  └──► pred (torch.float32, CUDA)
                                                   ▲
                                   Pano2stereo._depth_update_worker
                                                   │ (to CuPy via DLPack)
                                                   └──► latest_depth (CuPy)

Main loop: current_frame + latest_depth ──► stereo + repair ──► RTSP output
```

## 2. 关键组件与线程

- Thread: `FrameReader._read_loop()`
  - 解码输入流；每帧执行 BGR→RGB 与尺寸对齐（14 的倍数）。
  - 直接发送到 `FlashDepthProcessor.frame_queue`（避免主循环延迟影响时序）。
  - 线程安全更新 `current_frame`（`frame_lock`）。

- Thread: `FlashDepthProcessor.run_inference()`（外部帧模式）
  - 统一与流模式的推理逻辑，使用 `ExternalFrameDataset`；输入 batch 标准化为 `[B,1,C,H,W]`。
  - 发布 `self.pred = depth_pred.detach().contiguous().to(torch.float32)`，避免下游 bfloat16/DLPack 兼容性问题。
  - 支持 `max_frames` 结束条件，置位 `self.stopped`。

- Thread: `Pano2stereo._depth_update_worker()`
  - 若 `pred` 可用：在读取时完成 Torch(CUDA)→CuPy 的 DLPack 零拷贝，结果写入 `latest_depth`（受 `depth_lock` 保护）。
  - 高频更新（约 100fps），最大化主循环可用的最新深度。

- Thread: `Pano2stereo._stream_writer()`
  - 从 `_stream_queue` 异步取帧，写入 `ffmpeg` stdin，周期性 flush，降低主循环 I/O 等待。

- Main Thread（主循环）
  - 获取 `current_frame` 与 `latest_depth`（CuPy），进行立体图生成、遮挡修复、可选红青合成，并入队流输出。
  - 每 5 秒报告一次实际 FPS（maximum speed）。

## 3. 数据形状与类型约定

- 输入帧（FrameReader→FlashDepth）：`[C,H,W]` 浮点张量（来自 numpy HWC 的转换）。
- 模型输入（stream 模式）：`[B,1,C,H,W]`，T 固定为 1。
- 模型输出 `pred`：`torch.float32`（CUDA），发布前确保 `detach().contiguous().to(torch.float32)`。
- 深度缓存 `latest_depth`：CuPy 数组（DLPack 零拷贝完成于 `update_depth_from_flashdepth`）。
- 主循环中 RGB/Depth：均为 CuPy，避免 CPU/GPU 往返。

## 4. 性能与时序要点

- 单一解码：只由 `FrameReader` 解码一次，避免双流重复开销。
- 直接发送：帧读取后即时送入模型队列，深度预测及时性最佳。
- 无帧率限制：`FrameReader`、`run_inference`、主循环均不做人为 FPS 限制，按硬件极限运行。
- 零拷贝：Torch(CUDA)→CuPy 使用 DLPack，减少数据复制。
- 背景写流：将 RTSP I/O 移至后台线程。

## 5. 关键实现细节

- `pano2stereo.py`
  - `FrameReader._process_frame`: BGR→RGB；必要时 resize 至 14 倍数。
  - `FrameReader._send_to_flashdepth`: 将 HWC→CHW 的浮点张量送入 `frame_queue`（非阻塞，丢弃最旧）。
  - `Pano2stereo.update_depth_from_flashdepth`: 若 `pred` 可用，CUDA 张量 `detach().contiguous()` 后通过 `torch.utils.dlpack.to_dlpack()` 与 `cp.fromDlpack/cp.from_dlpack` 零拷贝为 CuPy，并存入 `latest_depth`。
  - 主循环：使用已转换好的 `latest_depth`（CuPy）；调用 `generate_stereo_pair`（内含 `_forward_map_zbuffer` 的 RawKernel 实现），随后 `repair_black_regions_pair` 进行一次性合并修复；可选 `gpu_create_anaglyph` 生成红青图并流式输出。
  - 错误兜底：如修复失败，则回退 `left_repaired/right_repaired = left/right`，避免未赋值异常。

- `submodule/Flashdepth/inference.py`
  - 外部帧模式与流模式共用一套推理流程：外部帧通过 `ExternalFrameDataset` 封装；所有输入标准化为 `[B,1,C,H,W]`。
  - 输出 `self.pred` 强制为 `float32`，规避 CuPy 对 `bfloat16` 的限制。

## 6. 运行与使用

- 配置：`configs/pano.yaml` 中设置 `URL`；`configs/flashdepth.yaml` 提供 FlashDepth 模型相关参数。
- 启动：
  ```bash
  python pano2stereo.py
  ```
- 输出：RTSP 推流至 `Pano2stereo.target_url`（代码默认值：`rtsp://10.20.35.30:28552/result`），并在 `output/run_xxx/visualization/stream_frames/` 周期性落盘帧。