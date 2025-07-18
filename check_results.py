#!/usr/bin/env python3
"""
检查优化结果脚本
"""
import os
import time
from pathlib import Path

def check_latest_results():
    """检查最新的处理结果"""
    results_dir = Path("results")
    
    print("🎯 Pano2Stereo 优化结果检查")
    print("=" * 50)
    
    if not results_dir.exists():
        print("❌ 没有找到结果目录")
        return
    
    # 找到最新的输出目录
    output_dirs = list(results_dir.glob("output_*"))
    if not output_dirs:
        print("❌ 没有找到输出文件")
        return
    
    latest_dir = max(output_dirs, key=os.path.getctime)
    files = list(latest_dir.glob("*"))
    
    print(f"📁 最新输出目录: {latest_dir}")
    print(f"🕒 创建时间: {time.ctime(os.path.getctime(latest_dir))}")
    print(f"📄 生成文件数: {len(files)}")
    
    print("\n📝 生成的文件:")
    for f in sorted(files):
        file_size = f.stat().st_size
        size_mb = file_size / (1024 * 1024)
        print(f"  ✅ {f.name} ({size_mb:.2f} MB)")
    
    # 检查核心文件
    expected_files = ["left.png", "right.png", "left_repaired.png", "right_repaired.png", "depth.png", "stereo.jpg"]
    missing_files = [f for f in expected_files if not (latest_dir / f).exists()]
    
    if not missing_files:
        print("\n🎉 所有核心文件都已生成!")
    else:
        print(f"\n⚠️  缺失文件: {missing_files}")
    
    # 检查日志
    log_file = Path("logs/pano2stereo.log")
    if log_file.exists() and log_file.stat().st_size > 0:
        print(f"\n📋 日志文件大小: {log_file.stat().st_size} bytes")
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 查找性能信息
        timing_lines = [line for line in lines if "TIMER" in line or "Core Processing" in line]
        if timing_lines:
            print("\n⏱️ 性能信息:")
            for line in timing_lines[-10:]:  # 最后10行计时信息
                print(f"  {line.strip()}")
    else:
        print("\n📋 日志文件为空或不存在")
    
    print(f"\n🏁 检查完成!")

if __name__ == "__main__":
    check_latest_results()
