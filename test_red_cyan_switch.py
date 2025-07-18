#!/usr/bin/env python3
"""
测试脚本：演示如何启用/禁用红蓝图生成功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pano2stereo_optimized import *

def test_with_red_cyan():
    """测试启用红蓝图功能"""
    print("="*60)
    print("测试：启用红蓝图功能")
    print("="*60)
    
    # 临时修改参数
    original_enable_red_cyan = PARAMS["ENABLE_RED_CYAN"]
    PARAMS["ENABLE_RED_CYAN"] = True
    
    try:
        main_optimized()
    finally:
        # 恢复原始设置
        PARAMS["ENABLE_RED_CYAN"] = original_enable_red_cyan

def test_without_red_cyan():
    """测试禁用红蓝图功能（默认）"""
    print("="*60)
    print("测试：禁用红蓝图功能（默认设置）")
    print("="*60)
    
    main_optimized()

def compare_performance():
    """性能对比测试"""
    print("\n" + "="*60)
    print("性能对比：红蓝图功能 开启 vs 关闭")
    print("="*60)
    
    print("\n🔴 测试1：关闭红蓝图功能（默认）")
    PARAMS["ENABLE_RED_CYAN"] = False
    main_optimized()
    
    print("\n🔵 测试2：开启红蓝图功能")
    PARAMS["ENABLE_RED_CYAN"] = True
    main_optimized()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试红蓝图功能开关')
    parser.add_argument('--mode', choices=['default', 'with_red_cyan', 'compare'], 
                       default='default', help='测试模式')
    
    args = parser.parse_args()
    
    if args.mode == 'default':
        test_without_red_cyan()
    elif args.mode == 'with_red_cyan':
        test_with_red_cyan()
    elif args.mode == 'compare':
        compare_performance()
