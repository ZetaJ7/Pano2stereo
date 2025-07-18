from PIL import Image
import os

def resize_image(input_path):
    # 定义目标分辨率字典（名称: (宽, 高)）
    resolutions = {
        '2K': (2560, 1440),
        '1080P': (1920, 1080),
        '720P': (1280, 720),
        '480P': (854, 480),
        '360P': (640, 360)
    }

    # 打开原始图像
    with Image.open(input_path) as img:
        # 获取基础文件名和扩展名
        base, ext = os.path.splitext(input_path)
        
        # 遍历所有分辨率进行重采样
        for res_name, dimensions in resolutions.items():
            # 使用LANCZOS算法进行高质量下采样
            resized_img = img.resize(dimensions, Image.Resampling.LANCZOS)
            
            # 构建输出文件名
            output_path = f"{base}_{res_name}{ext}"
            
            # 保存图像（保留PNG格式的元数据）
            resized_img.save(output_path, format='PNG', optimize=True)
            print(f"已生成: {os.path.basename(output_path)}")

if __name__ == "__main__":
    input_image = "Lab1.png"  # 修改为你的图片路径
    resize_image(input_image)