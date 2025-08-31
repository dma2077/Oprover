#!/usr/bin/env python3
"""
从所有parquet文件中下载GitHub文件 (简化版)
- 采用纯多线程模型，逐个处理Parquet文件。
- 支持按文件组(group)下载。
- 支持两级断点续下：
  1. 跳过已完全处理的Parquet文件。
  2. 在处理单个Parquet文件时，跳过已存在的代码文件。
- 使用tqdm显示每个Parquet文件的详细下载进度。
"""

import os
import pandas as pd
import requests
import time
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import glob
import math

# ==============================================================================
# 1. 核心下载函数
# ==============================================================================

def download_file_from_github(repo, commit_id, rel_path, output_dir):
    """
    从GitHub下载单个文件。
    内置文件存在性检查，这是实现底层断点续下的关键。

    Args:
        repo (str): 仓库名称
        commit_id (str): 提交ID
        rel_path (str): 文件在仓库中的相对路径
        output_dir (str): 保存该文件的目录

    Returns:
        dict: 包含下载结果的字典
    """
    try:
        # 构建GitHub Raw文件URL
        url = f"https://raw.githubusercontent.com/{repo}/{commit_id}/{rel_path}"

        # 根据相对路径创建子目录
        file_subdirectory = os.path.dirname(rel_path)
        full_dir_path = os.path.join(output_dir, file_subdirectory)
        Path(full_dir_path).mkdir(parents=True, exist_ok=True)

        # 构建最终输出文件的完整路径
        filename = os.path.basename(rel_path)
        output_path = os.path.join(full_dir_path, filename)

        # 【断点续下关键点2】如果文件已存在，则跳过下载
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            return {'success': True, 'skipped': True, 'size': size, 'error': None}

        # 发起下载请求
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # 如果HTTP状态码是4xx或5xx，则抛出异常

        # 将下载内容写入文件
        with open(output_path, 'wb') as f:
            f.write(response.content)

        size = len(response.content)
        return {'success': True, 'skipped': False, 'size': size, 'error': None}

    except Exception as e:
        return {'success': False, 'skipped': False, 'size': 0, 'error': str(e)}

def download_worker(task):
    """
    供线程池调用的工作函数，解包参数并调用核心下载函数。
    
    Args:
        task (tuple): (repo, commit_id, rel_path, output_dir)
    
    Returns:
        dict: 包含原始任务信息和下载结果的字典
    """
    repo, commit_id, rel_path, output_dir = task
    result = download_file_from_github(repo, commit_id, rel_path, output_dir)
    # 将任务信息附加到结果中，便于后续追踪
    result.update({'repo': repo, 'rel_path': rel_path})
    return result

# ==============================================================================
# 2. Parquet 文件处理函数
# ==============================================================================

def process_single_parquet_file(file_path, base_output_dir, max_threads):
    """
    使用多线程下载单个Parquet文件中列出的所有文件。

    Args:
        file_path (str): Parquet文件的路径。
        base_output_dir (str): 下载文件的根目录。
        max_threads (int): 用于下载的最大线程数。

    Returns:
        dict: 处理结果的统计信息。
    """
    try:
        parquet_filename = os.path.splitext(os.path.basename(file_path))[0]
        current_output_dir = os.path.join(base_output_dir, parquet_filename)
        Path(current_output_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n--- 开始处理 Parquet 文件: {os.path.basename(file_path)} ---")
        print(f"    - 文件将保存到: {current_output_dir}")
        print(f"    - 使用线程数: {max_threads}")

        df = pd.read_parquet(file_path)
        total_records = len(df)
        if total_records == 0:
            print("    - 文件为空，跳过。")
            return {'success': True, 'total_files': 0, 'downloaded': 0, 'skipped': 0, 'failed': 0}

        # 创建所有下载任务的列表
        tasks = [
            (row['repo'], row['commit_id'], row['rel_path'], current_output_dir)
            for _, row in df.iterrows()
        ]

        # 初始化统计计数器
        stats = {'downloaded': 0, 'skipped': 0, 'failed': 0, 'total_size': 0}

        # 使用线程池和tqdm进度条执行下载
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            # 提交所有任务
            future_to_task = {executor.submit(download_worker, task): task for task in tasks}

            # 使用tqdm包装as_completed来显示进度
            pbar = tqdm(as_completed(future_to_task), total=total_records, desc="下载进度", unit="文件")
            for future in pbar:
                result = future.result()

                if result['success']:
                    stats['total_size'] += result['size']
                    if result['skipped']:
                        stats['skipped'] += 1
                    else:
                        stats['downloaded'] += 1
                else:
                    stats['failed'] += 1
                
                # 更新tqdm的后缀信息
                pbar.set_postfix({
                    '下载': stats['downloaded'],
                    '跳过': stats['skipped'],
                    '失败': stats['failed']
                })

        print(f"--- Parquet 文件处理完成: {os.path.basename(file_path)} ---")
        print(f"    - 总记录数: {total_records:,}")
        print(f"    - 新下载文件: {stats['downloaded']:,}")
        print(f"    - 已存在跳过: {stats['skipped']:,}")
        print(f"    - 下载失败: {stats['failed']:,}")
        print(f"    - 总大小: {stats['total_size'] / 1024 / 1024:.2f} MB")
        
        return {'success': True, **stats}

    except Exception as e:
        print(f"\n处理文件 {os.path.basename(file_path)} 时发生严重错误: {e}")
        return {'success': False, 'error': str(e)}

# ==============================================================================
# 3. 顶层控制和断点续下逻辑
# ==============================================================================

def get_completed_files_path(output_dir, group_id):
    """获取记录已完成Parquet文件列表的路径"""
    return os.path.join(output_dir, f"_completed_parquet_group_{group_id:04d}.txt")

def load_completed_files(completed_files_path):
    """加载已完成的Parquet文件列表"""
    if not os.path.exists(completed_files_path):
        return set()
    try:
        with open(completed_files_path, 'r', encoding='utf-8') as f:
            return {line.strip() for line in f if line.strip()}
    except Exception as e:
        print(f"警告: 加载已完成文件列表失败: {e}")
        return set()

def save_completed_file(completed_files_path, parquet_filename):
    """将已完成的Parquet文件名添加到完成列表"""
    try:
        with open(completed_files_path, 'a', encoding='utf-8') as f:
            f.write(f"{parquet_filename}\n")
    except Exception as e:
        print(f"警告: 保存已完成文件记录失败: {e}")

def download_group_files(input_dir, output_dir, group_id, max_threads):
    """
    下载指定文件组（group_id）的所有文件。
    该函数逐个处理Parquet文件。
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_parquet_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not all_parquet_files:
        print(f"错误: 输入目录 '{input_dir}' 中未找到任何 .parquet 文件。")
        return

    # 每个组包含50个文件
    files_per_group = 50
    total_groups = math.ceil(len(all_parquet_files) / files_per_group)

    if group_id >= total_groups:
        print(f"错误: 块ID {group_id} 超出范围。总共有 {total_groups} 个块 (ID从 0 到 {total_groups-1})。")
        return

    start_index = group_id * files_per_group
    end_index = min(start_index + files_per_group, len(all_parquet_files))
    group_files_to_process = all_parquet_files[start_index:end_index]

    print(f"总共找到 {len(all_parquet_files)} 个 Parquet 文件, 分为 {total_groups} 个块。")
    print(f"当前任务: 下载块 {group_id} (包含 {len(group_files_to_process)} 个 Parquet 文件)")

    # 【断点续下关键点1】加载已完成的Parquet文件列表
    completed_files_path = get_completed_files_path(output_dir, group_id)
    completed_files = load_completed_files(completed_files_path)
    print(f"已加载 {len(completed_files)} 个已完成的 Parquet 文件记录。")

    remaining_files = [
        f for f in group_files_to_process if os.path.basename(f) not in completed_files
    ]
    
    if not remaining_files:
        print(f"\n块 {group_id} 中的所有 Parquet 文件均已处理完毕！")
        return

    print(f"块 {group_id} 中还有 {len(remaining_files)} 个 Parquet 文件待处理。")
    start_time = time.time()

    # 依次处理每个待处理的Parquet文件
    for i, file_path in enumerate(remaining_files):
        print(f"\n{'='*80}")
        print(f"开始处理块内文件 {i+1}/{len(remaining_files)}: {os.path.basename(file_path)}")
        print(f"{'='*80}")
        
        result = process_single_parquet_file(file_path, output_dir, max_threads)

        if result.get('success'):
            # 处理成功后，记录到完成列表
            save_completed_file(completed_files_path, os.path.basename(file_path))
            print(f"✓ Parquet 文件 {os.path.basename(file_path)} 已成功处理并标记为完成。")
        else:
            print(f"✗ Parquet 文件 {os.path.basename(file_path)} 处理失败，将继续处理下一个文件。")

    end_time = time.time()
    total_duration_minutes = (end_time - start_time) / 60
    print(f"\n{'*'*80}")
    print(f"块 {group_id} 全部任务处理完成! 总耗时: {total_duration_minutes:.2f} 分钟")
    print(f"{'*'*80}")

# ==============================================================================
# 4. 主程序入口
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='从Parquet文件中下载GitHub文件 (纯多线程，逐个文件处理)。',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input_dir', default='/madehua/data/Nemotron-Pretraining-Code-v1/Nemotron-Code-Metadata',
                        help='包含Parquet文件的输入目录路径')
    parser.add_argument('--output_dir', default='/mnt/oprover-data',
                        help='保存下载文件的输出目录路径')
    parser.add_argument('--threads', type=int, default=32,
                        help='下载线程数 (默认: 32)')
    parser.add_argument('--group_id', type=int, required=True,
                        help='要下载的文件块ID (从0开始，每个块默认50个Parquet文件)')
    
    args = parser.parse_args()
    
    print("下载任务启动参数:")
    print(f"  - 输入目录: {args.input_dir}")
    print(f"  - 输出目录: {args.output_dir}")
    print(f"  - 下载线程数: {args.threads}")
    print(f"  - 目标块ID: {args.group_id}")
    print(f"="*60)
    
    download_group_files(args.input_dir, args.output_dir, args.group_id, args.threads)

if __name__ == "__main__":
    main()