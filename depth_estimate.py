import os,sys
import cv2
import torch
import numpy as np
import logging

from submodule.DepthAnythingv2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
# from submodule.DepthAnythingv2.depth_anything_v2.dpt import DepthAnythingV2

def depth_estimation(raw_img, max_depth=20,encoder='vitl', dataset='hypersim'):
    # input : cv2.imread
    # output : numpy array of depth map in meters

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    logging.info(f'Using device: {DEVICE}')

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}, 
    }

    encoder = encoder # or 'vits', 'vitb', 'vitl'
    dataset = dataset # 'hypersim' for indoor model, 'vkitti' for outdoor model

    logging.info(f'Loading {dataset} model with {encoder} encoder...')
    t3 = cv2.getTickCount()
    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth}) # max_depth: 20 for indoor model, 80 for outdoor model
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=DEVICE))
    model = model.to(DEVICE).eval()
    t4 = cv2.getTickCount()
    logging.info(f'[DepthAnythingV2] Model loading time: {(t4 - t3) / cv2.getTickFrequency()} seconds')
    
    t1 = cv2.getTickCount()
    depth = model.infer_image(raw_img) # HxW depth map in meters in numpy
    t2 = cv2.getTickCount()
    logging.info(f'[DepthAnythingV2] Inference time: {(t2 - t1) / cv2.getTickFrequency()} seconds')

    return depth

def main():
    raw_img = cv2.imread('Panoramic.png')
    depth = depth_estimation(raw_img, max_depth=20) # max_depth: 20 for indoor model, 80 for outdoor model
    np.save('depth_data.npy', depth)
    min,max = depth.min(), depth.max()
    logging.info(f'Depth min: {min}, max: {max}')
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)
    cv2.imwrite('depth_uint8.png', depth_uint8)  
    cv2.imshow('Depth Map (Grayscale)', depth_uint8)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()