#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练日志Loss曲线绘制工具
解析log文件夹中的训练日志，提取loss数据并绘制曲线图
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_file(log_path):
    """
    解析单个日志文件，提取epoch和loss数据
    
    Args:
        log_path: 日志文件路径
        
    Returns:
        tuple: (epochs, losses, model_name)
    """
    epochs = []
    losses = []
    
    # 从文件名提取模型名称
    filename = Path(log_path).stem
    # 移除时间戳部分，保留模型配置信息
    model_name = '_'.join(filename.split('_')[:-6])  # 移除日期时间部分
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 使用正则表达式匹配epoch和loss信息
        pattern = r'Epoch---(\d+),\s*total_loss---([0-9.]+)'
        matches = re.findall(pattern, content)
        
        for epoch_str, loss_str in matches:
            epochs.append(int(epoch_str))
            losses.append(float(loss_str))
            
    except Exception as e:
        print(f"解析文件 {log_path} 时出错: {e}")
        return [], [], model_name
    
    return epochs, losses, model_name

def plot_loss_curves():
    """
    绘制所有日志文件的loss曲线
    """
    log_dir = Path('log')
    
    # 获取所有.txt日志文件
    log_files = list(log_dir.glob('*.txt'))
    
    if not log_files:
        print("未找到日志文件！")
        return
    
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Loss Curves Comparison', fontsize=16, fontweight='bold')
    
    # 颜色列表
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # 按数据集分组
    irstd_logs = []
    nudt_logs = []
    
    for log_file in log_files:
        if 'IRSTD-1K' in log_file.name:
            irstd_logs.append(log_file)
        elif 'NUDT-SIRST' in log_file.name:
            nudt_logs.append(log_file)
    
    # 绘制IRSTD-1K数据集的结果
    ax1 = axes[0, 0]
    ax1.set_title('IRSTD-1K Dataset Loss Curves', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.grid(True, alpha=0.3)
    
    for i, log_file in enumerate(irstd_logs):
        epochs, losses, model_name = parse_log_file(log_file)
        if epochs and losses:
            color = colors[i % len(colors)]
            ax1.plot(epochs, losses, color=color, linewidth=2, 
                    label=model_name.replace('IRSTD-1K_', ''), marker='o', markersize=3)
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 绘制NUDT-SIRST数据集的结果
    ax2 = axes[0, 1]
    ax2.set_title('NUDT-SIRST Dataset Loss Curves', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Total Loss')
    ax2.grid(True, alpha=0.3)
    
    for i, log_file in enumerate(nudt_logs):
        epochs, losses, model_name = parse_log_file(log_file)
        if epochs and losses:
            color = colors[i % len(colors)]
            ax2.plot(epochs, losses, color=color, linewidth=2,
                    label=model_name.replace('NUDT-SIRST_', ''), marker='o', markersize=3)
    
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 绘制所有模型的对比图
    ax3 = axes[1, 0]
    ax3.set_title('All Models Loss Curves Comparison', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Total Loss')
    ax3.grid(True, alpha=0.3)
    
    all_logs = irstd_logs + nudt_logs
    for i, log_file in enumerate(all_logs):
        epochs, losses, model_name = parse_log_file(log_file)
        if epochs and losses:
            color = colors[i % len(colors)]
            ax3.plot(epochs, losses, color=color, linewidth=1.5, 
                    label=model_name, alpha=0.8)
    
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 统计信息
    ax4 = axes[1, 1]
    ax4.set_title('Training Statistics', fontweight='bold')
    ax4.axis('off')
    
    stats_text = "Training Log Statistics:\n\n"
    stats_text += f"Total log files: {len(log_files)}\n"
    stats_text += f"IRSTD-1K dataset: {len(irstd_logs)} models\n"
    stats_text += f"NUDT-SIRST dataset: {len(nudt_logs)} models\n\n"
    
    # 计算每个模型的最终loss
    stats_text += "Final Loss Values:\n"
    for log_file in all_logs:
        epochs, losses, model_name = parse_log_file(log_file)
        if epochs and losses:
            final_loss = losses[-1]
            stats_text += f"{model_name}: {final_loss:.6f}\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = 'loss_curves_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Loss curves saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    plot_loss_curves()