# ================== Pano2Stereo 配置文件 ==================
# 这个文件包含了程序的所有可配置参数
# 修改这里的参数来调整程序行为

# ============== 基础立体视觉参数 ==============
IPD = 0.032                         # 瞳距 (米) - 人眼间距，影响立体效果强度
PRO_RADIUS = 1                      # 视环投影半径 (米)
PIXEL = 4448                        # 赤道图像宽度 (像素)
CRITICAL_DEPTH = 9999               # 临界深度 (米) - 超过此距离视为无穷远

# ============== 输入输出设置 ==============
IMG_PATH = "Data/Lab3.jpg"          # 输入图片路径
OUTPUT_BASE = "results/output"      # 输出目录基础名
GENERATE_VIDEO = False              # 是否生成视频文件

# ============== 深度估计模型设置 ==============
VIT_SIZE = "vits"                   # 深度模型类型: "vits"(最快), "vitb"(中等), "vitl"(最慢但最准确)
DATASET = "hypersim"                # 深度模型数据集: "hypersim" 或 "vkitti"

# ============== 性能优化设置 ==============
TARGET_HEIGHT = 540                 # 目标图像高度 (像素) - 降低以提升速度
TARGET_WIDTH = 960                  # 目标图像宽度 (像素) - 降低以提升速度

# ============== 功能开关 ==============
ENABLE_REPAIR = True                # 是否启用黑色区域修复
ENABLE_POST_PROCESS = True          # 是否启用后处理
ENABLE_RED_CYAN = False             # 是否生成红青3D图 (关闭可提升约0.4秒速度)

# ============== 性能预设 ==============
# 取消注释下面的预设之一来快速配置性能参数

# 🚀 极速模式 (约1秒完成)
# TARGET_HEIGHT = 270
# TARGET_WIDTH = 480
# ENABLE_REPAIR = False
# ENABLE_RED_CYAN = False
# VIT_SIZE = "vits"

# ⚡ 快速模式 (约1.5秒完成)
# TARGET_HEIGHT = 540
# TARGET_WIDTH = 960
# ENABLE_REPAIR = True
# ENABLE_RED_CYAN = False
# VIT_SIZE = "vits"

# 🎯 平衡模式 (约2秒完成)
# TARGET_HEIGHT = 720
# TARGET_WIDTH = 1280
# ENABLE_REPAIR = True
# ENABLE_RED_CYAN = True
# VIT_SIZE = "vits"

# 🏆 质量模式 (约5秒完成)
# TARGET_HEIGHT = 1080
# TARGET_WIDTH = 1920
# ENABLE_REPAIR = True
# ENABLE_RED_CYAN = True
# VIT_SIZE = "vitb"
