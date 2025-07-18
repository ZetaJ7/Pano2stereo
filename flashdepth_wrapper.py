"""
FlashDepth 深度估计模块包装器
用于集成到 pano2stereo_optimized 中
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path

# 添加FlashDepth到Python路径
current_dir = Path(__file__).parent
flashdepth_dir = current_dir / "submodule" / "Flashdepth"
sys.path.insert(0, str(flashdepth_dir))

from flashdepth.model import FlashDepth
import yaml

class FlashDepthWrapper:
    """FlashDepth模型包装器，提供简单的推理接口"""
    
    def __init__(self, model_type='flashdepth-l', device='cuda'):
        """
        初始化FlashDepth模型
        
        Args:
            model_type: 模型类型 ('flashdepth', 'flashdepth-l', 'flashdepth-s')
            device: 设备类型 ('cuda', 'cpu')
        """
        self.device = device
        self.model_type = model_type
        
        # 模型配置路径
        self.config_dir = flashdepth_dir / "configs" / model_type
        self.config_path = self.config_dir / "config.yaml"
        
        # 加载配置
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 设置模型参数
        vit_size = self.config['model']['vit_size']
        patch_size = self.config['model']['patch_size']
        use_mamba = self.config['model']['use_mamba']
        
        # 创建模型
        model_kwargs = {
            'use_mamba': use_mamba,
            'mamba_type': self.config['model']['mamba_type'],
            'num_mamba_layers': self.config['model']['num_mamba_layers'],
            'downsample_mamba': self.config['model']['downsample_mamba'],
            'mamba_in_dpt_layer': self.config['model']['mamba_in_dpt_layer'],
            'training': False,  # 推理模式
            'mamba_d_conv': self.config['model'].get('mamba_d_conv', 4),
            'mamba_d_state': self.config['model'].get('mamba_d_state', 256),
            'mamba_pos_embed': self.config['model'].get('mamba_pos_embed', None),
            'hybrid_configs': self.config.get('hybrid_configs', None),
        }
        
        self.model = FlashDepth(
            vit_size=vit_size,
            patch_size=patch_size,
            **model_kwargs
        )
        
        # 加载预训练权重
        checkpoint_path = self._find_checkpoint()
        if checkpoint_path.exists():
            logging.info(f"Loading FlashDepth checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 处理checkpoint格式
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
                
            self.model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # 移动到设备并设置为评估模式
        self.model = self.model.to(device).eval()
        
        logging.info(f"FlashDepth model loaded: {model_type}")
    
    def _find_checkpoint(self):
        """查找预训练模型文件"""
        # 检查config.yaml中指定的路径
        if self.config.get('load'):
            checkpoint_path = Path(self.config['load'])
            if checkpoint_path.is_absolute():
                return checkpoint_path
            else:
                return self.config_dir / checkpoint_path
        
        # 查找默认的checkpoint文件
        for pattern in ['*.pth', '*.pt']:
            checkpoints = list(self.config_dir.glob(pattern))
            if checkpoints:
                return checkpoints[0]
        
        raise FileNotFoundError(f"No checkpoint found in {self.config_dir}")
    
    def infer_image(self, image):
        """
        对单张图像进行深度估计
        
        Args:
            image: 输入图像，numpy数组 (H, W, 3) 或 (H, W, 3) uint8
            
        Returns:
            depth: 深度图，numpy数组 (H, W)
        """
        # 预处理输入图像
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
        
        # 转换为torch tensor
        if len(image.shape) == 3:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        # 添加时间维度用于视频处理 (B, T, C, H, W)
        image_tensor = image_tensor.unsqueeze(1)  # (1, 1, 3, H, W)
        image_tensor = image_tensor.to(self.device)
        
        # 推理
        with torch.no_grad():
            # 如果使用mamba，需要重置序列
            if self.model.use_mamba:
                self.model.mamba.start_new_sequence()
            
            # 获取单帧
            frame = image_tensor[:, 0, :, :, :]  # (1, 3, H, W)
            B, C, H, W = frame.shape
            
            patch_h, patch_w = H // self.model.patch_size, W // self.model.patch_size
            
            # 获取DPT特征
            dpt_features = self.model.get_dpt_features(frame, input_shape=(B, C, H, W))
            
            # 最终预测
            pred_depth = self.model.final_head(dpt_features, patch_h, patch_w)
            pred_depth = torch.clip(pred_depth, min=0)
            
            # 转换为numpy
            depth_np = pred_depth.squeeze().cpu().numpy()
        
        return depth_np
    
    def infer_video(self, video_frames):
        """
        对视频帧序列进行深度估计
        
        Args:
            video_frames: 视频帧列表，每帧为numpy数组 (H, W, 3)
            
        Returns:
            depths: 深度图列表，每个为numpy数组 (H, W)
        """
        # 预处理视频帧
        processed_frames = []
        for frame in video_frames:
            if isinstance(frame, np.ndarray):
                if frame.dtype == np.uint8:
                    frame = frame.astype(np.float32) / 255.0
            processed_frames.append(frame)
        
        # 转换为torch tensor (B, T, C, H, W)
        video_tensor = torch.stack([
            torch.from_numpy(frame).permute(2, 0, 1) 
            for frame in processed_frames
        ]).unsqueeze(0)  # (1, T, 3, H, W)
        
        video_tensor = video_tensor.to(self.device)
        
        # 推理
        with torch.no_grad():
            # 使用模型的forward方法
            dummy_gif_path = "/tmp/dummy.gif"
            result = self.model.forward(
                video_tensor,
                use_mamba=self.model.use_mamba,
                gif_path=dummy_gif_path,
                resolution=512,
                out_mp4=False,
                save_depth_npy=False,
                print_time=False
            )
            
            # 提取深度预测
            if isinstance(result, tuple):
                depths = result[1]  # 假设深度在第二个位置
            else:
                depths = result
            
            # 转换为numpy列表
            depth_list = [depth.squeeze().cpu().numpy() for depth in depths]
        
        return depth_list


def create_flashdepth_model(model_type='flashdepth-l', device='cuda'):
    """
    创建FlashDepth模型的便捷函数
    
    Args:
        model_type: 模型类型 ('flashdepth', 'flashdepth-l', 'flashdepth-s')
        device: 设备类型
        
    Returns:
        FlashDepthWrapper实例
    """
    return FlashDepthWrapper(model_type=model_type, device=device)
