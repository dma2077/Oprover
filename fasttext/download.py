#!/usr/bin/env python3
"""
从所有parquet文件中下载GitHub文件
使用多进程+多线程和进度条，按1000个文件分组到不同目录
支持断点续下和分块下载
"""

import os
import pandas as pd
import requests
import time
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import re
import glob
import math
from collections import defaultdict
import threading
import json
import pickle
import multiprocessing as mp
from functools import partial

# 线程锁，用于保护共享资源
lock = threading.Lock()

# ... existing code ...

def process_single_parquet_file_mp(args):
    """
    多进程版本的单个parquet文件处理函数
    
    Args:
        args: 包含所有参数的元组
    
    Returns:
        dict: 处理结果
    """
    # 解包参数
    file_path, output_base_dir, max_workers, checkpoint_data, group_id, file_index_in_group = args
    
    try:
        print(f"\n[进程 {os.getpid()}] 开始处理文件: {os.path.basename(file_path)} (块 {group_id}, 文件 {file_index_in_group})")
        
        # 读取parquet文件
        df = pd.read_parquet(file_path)
        total_files = len(df)
        
        print(f"[进程 {os.getpid()}] 文件包含 {total_files:,} 条记录")
        print(f"[进程 {os.getpid()}] 使用 {max_workers} 个线程进行下载")
        
        # 检查点键
        checkpoint_key = f"group_{group_id:04d}_file_{file_index_in_group:04d}"
        
        # 从检查点恢复状态
        if checkpoint_data and checkpoint_key in checkpoint_data:
            resume_info = checkpoint_data[checkpoint_key]
            start_index = resume_info.get('completed_files', 0)
            print(f"[进程 {os.getpid()}] 从检查点恢复，已完成 {start_index:,} 个文件")
        else:
            start_index = 0
            print(f"[进程 {os.getpid()}] 开始新的下载任务")
        
        # 创建以parquet文件名命名的目录
        parquet_filename = os.path.splitext(os.path.basename(file_path))[0]
        current_dir = os.path.join(output_base_dir, parquet_filename)
        
        # 创建目录
        Path(current_dir).mkdir(parents=True, exist_ok=True)
        print(f"[进程 {os.getpid()}] 文件将保存到目录: {current_dir}")
        
        # 统计信息
        successful_downloads = start_index
        failed_downloads = 0
        total_size = 0
        
        # 准备下载任务（跳过已完成的）
        download_tasks = []
        for idx, row in df.iterrows():
            if idx < start_index:
                continue
            
            download_tasks.append((
                row['repo'], 
                row['commit_id'], 
                row['rel_path'], 
                current_dir,
                idx
            ))
        
        # 使用多线程下载
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有下载任务
            future_to_task = {executor.submit(download_single_file_worker, task): task for task in download_tasks}
            
            # 使用tqdm显示下载进度
            with tqdm(total=total_files, initial=start_index, desc=f"[进程{os.getpid()}] 下载进度", unit="文件") as pbar:
                for future in as_completed(future_to_task):
                    result = future.result()
                    
                    # 更新统计信息
                    if result['success'] and result['filename']:
                        successful_downloads += 1
                        total_size += result['size']
                    else:
                        failed_downloads += 1
                    
                    # 更新进度条
                    pbar.update(1)
                    pbar.set_postfix({
                        'Repo': result['repo'][:20] + '...' if len(result['repo']) > 20 else result['repo'],
                        'File': result['rel_path'][:30] + '...' if len(result['rel_path']) > 30 else result['rel_path'],
                        '成功': successful_downloads,
                        '失败': failed_downloads,
                        '线程': max_workers
                    })
                    
                    # 定期保存检查点
                    if successful_downloads % 100 == 0:
                        checkpoint_data[checkpoint_key] = {
                            'completed_files': successful_downloads,
                            'failed_files': failed_downloads,
                            'total_size': total_size,
                            'last_update': time.time()
                        }
        
        print(f"\n[进程 {os.getpid()}] 文件 {os.path.basename(file_path)} 处理完成:")
        print(f"  成功下载: {successful_downloads:,}")
        print(f"  下载失败: {failed_downloads:,}")
        print(f"  总大小: {total_size/1024/1024:.2f} MB")
        print(f"  使用线程数: {max_workers}")
        print(f"  保存目录: {current_dir}")
        
        return {
            'filename': os.path.basename(file_path),
            'total_files': total_files,
            'successful_downloads': successful_downloads,
            'failed_downloads': failed_downloads,
            'total_size': total_size,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        print(f"\n[进程 {os.getpid()}] 处理文件 {os.path.basename(file_path)} 时出错: {str(e)}")
        return {
            'filename': os.path.basename(file_path),
            'total_files': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_size': 0,
            'success': False,
            'error': str(e)
        }

def download_group_files_mp(input_dir, output_dir, group_id, max_workers=16, max_processes=None):
    """
    使用多进程+多线程下载指定块ID的parquet文件
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        group_id: 块ID (从0开始)
        max_workers: 每个进程的最大线程数
        max_processes: 最大进程数，默认为CPU核心数
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有parquet文件
    parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    parquet_files.sort()
    
    if not parquet_files:
        print("未找到parquet文件")
        return
    
    total_files = len(parquet_files)
    files_per_group = 50
    
    # 计算块信息
    total_groups = math.ceil(total_files / files_per_group)
    
    if group_id >= total_groups:
        print(f"错误: 块ID {group_id} 超出范围，总共有 {total_groups} 个块")
        return
    
    # 计算当前块的文件范围
    start_file_index = group_id * files_per_group
    end_file_index = min(start_file_index + files_per_group, total_files)
    
    current_group_files = parquet_files[start_file_index:end_file_index]
    
    print(f"找到 {total_files} 个parquet文件，分为 {total_groups} 个块")
    print(f"当前下载块 {group_id} (共 {total_groups} 个块)")
    print(f"块 {group_id} 包含文件 {start_file_index + 1} 到 {end_file_index}")
    print(f"输出目录: {output_dir}")
    print(f"每个进程线程数: {max_workers}")
    
    # 设置进程数
    if max_processes is None:
        max_processes = min(mp.cpu_count(), len(current_group_files))
    
    print(f"使用进程数: {max_processes}")
    print(f"每个parquet文件将创建对应的同名目录")
    print(f"="*60)
    
    # 检查点文件路径
    checkpoint_path = get_checkpoint_path(output_dir, group_id)
    
    # 已完成文件列表路径
    completed_files_path = get_completed_files_path(output_dir, group_id)
    
    # 加载检查点
    checkpoint_data = load_checkpoint(checkpoint_path)
    if checkpoint_data is None:
        checkpoint_data = {}
    
    # 加载已完成文件列表
    completed_files = load_completed_files(completed_files_path)
    
    # 过滤掉已完成的文件
    remaining_files = []
    for file_path in current_group_files:
        parquet_filename = os.path.basename(file_path)
        if parquet_filename not in completed_files:
            remaining_files.append(file_path)
        else:
            print(f"跳过已完成的文件: {parquet_filename}")
    
    if not remaining_files:
        print(f"块 {group_id} 中的所有文件都已完成下载！")
        return
    
    print(f"块 {group_id} 中还有 {len(remaining_files)} 个文件需要下载")
    print(f"="*60)
    
    # 准备多进程参数
    process_args = []
    for i, file_path in enumerate(remaining_files):
        file_index_in_group = start_file_index + i
        process_args.append((
            file_path, 
            output_dir, 
            max_workers,
            checkpoint_data,
            group_id,
            i
        ))
    
    # 使用多进程处理文件
    start_time = time.time()
    total_successful = 0
    total_failed = 0
    total_size = 0
    
    # 分批处理，避免创建过多进程
    batch_size = max_processes
    for batch_start in range(0, len(process_args), batch_size):
        batch_end = min(batch_start + batch_size, len(process_args))
        current_batch = process_args[batch_start:batch_end]
        
        print(f"\n处理批次 {batch_start//batch_size + 1}: 文件 {batch_start + 1} 到 {batch_end}")
        
        with ProcessPoolExecutor(max_workers=len(current_batch)) as executor:
            # 提交批次任务
            future_to_process = {executor.submit(process_single_parquet_file_mp, args): args for args in current_batch}
            
            # 收集结果
            for future in as_completed(future_to_process):
                result = future.result()
                args = future_to_process[future]
                file_path = args[0]
                parquet_filename = os.path.basename(file_path)
                
                # 更新总统计
                if result['success']:
                    total_successful += result['successful_downloads']
                    total_failed += result['failed_downloads']
                    total_size += result['total_size']
                    
                    # 标记该parquet文件为已完成
                    save_completed_file(completed_files_path, parquet_filename)
                    print(f"✓ 文件 {parquet_filename} 已完成，已记录到完成列表")
                else:
                    print(f"✗ 文件 {parquet_filename} 处理失败: {result.get('error', '未知错误')}")
                
                # 保存检查点
                save_checkpoint(checkpoint_path, checkpoint_data)
    
    end_time = time.time()
    
    # 显示结果
    print(f"\n" + "="*60)
    print(f"块 {group_id} 下载完成! 总耗时: {end_time - start_time:.2f} 秒")
    print(f"="*60)
    print(f"块ID: {group_id}")
    print(f"总文件数: {len(current_group_files)}")
    print(f"已完成文件: {len(completed_files)}")
    print(f"本次处理文件: {len(remaining_files)}")
    print(f"总成功下载: {total_successful:,}")
    print(f"总下载失败: {total_failed:,}")
    print(f"总大小: {total_size/1024/1024:.2f} MB")
    print(f"每个进程线程数: {max_workers}")
    print(f"使用进程数: {max_processes}")
    print(f"输出目录: {output_dir}")
    print(f"检查点文件: {checkpoint_path}")
    print(f"已完成文件列表: {completed_files_path}")
    print(f"="*60)

def main():
    parser = argparse.ArgumentParser(description='从parquet文件中下载GitHub文件，支持多进程+多线程分块下载和断点续下')
    parser.add_argument('--input_dir', default='/madehua/data/Nemotron-Pretraining-Code-v1/Nemotron-Code-Metadata',
                       help='输入目录路径')
    parser.add_argument('--output_dir', default='/mnt/oprover-data',
                       help='输出目录路径')
    parser.add_argument('--threads', type=int, default=16,
                       help='每个进程的下载线程数 (默认: 16)')
    parser.add_argument('--processes', type=int, default=None,
                       help='最大进程数 (默认: CPU核心数)')
    parser.add_argument('--group_id', type=int, required=True,
                       help='块ID (从0开始，每个块包含50个parquet文件)')
    
    args = parser.parse_args()
    
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"每个进程线程数: {args.threads}")
    print(f"最大进程数: {args.processes or 'CPU核心数'}")
    print(f"块ID: {args.group_id}")
    print(f"="*60)
    
    # 使用多进程+多线程下载指定块的文件
    download_group_files_mp(args.input_dir, args.output_dir, args.group_id, args.threads, args.processes)

if __name__ == "__main__":
    main()