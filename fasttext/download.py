#!/usr/bin/env python3
"""
从所有parquet文件中下载GitHub文件
使用多进程+多线程和进度条，按50个文件分组到不同目录
支持断点续下和分块下载

修改版：
- 实现了真正的进程间数据分片：将单个Parquet文件中的任务分配给多个进程。
- 简化了断点续下机制，使其在多进程环境下更可靠。
- 优化了代码结构和日志输出，逻辑更清晰。
"""

import os
import pandas as pd
import requests
import time
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import glob
import math
import json
import multiprocessing as mp

# ==============================================================================
# 1. 单文件下载核心函数 (最底层工作单元)
# ==============================================================================

def download_file_from_github(repo, commit_id, rel_path, output_dir):
    """
    从GitHub下载单个文件。
    内置了文件存在性检查，这是实现断点续下的关键。

    Args:
        repo (str): 仓库名称
        commit_id (str): 提交ID
        rel_path (str): 相对路径
        output_dir (str): 输出目录

    Returns:
        dict: 下载结果
    """
    try:
        # 构建GitHub Raw URL
        url = f"https://raw.githubusercontent.com/{repo}/{commit_id}/{rel_path}"

        # 创建输出目录
        file_dir = os.path.join(output_dir, os.path.dirname(rel_path))
        Path(file_dir).mkdir(parents=True, exist_ok=True)

        # 构建输出文件路径
        filename = os.path.basename(rel_path)
        output_path = os.path.join(file_dir, filename)

        # 如果文件已存在，直接成功返回，实现文件级别的断点续下
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            return {'success': True, 'skipped': True, 'size': size, 'error': None}

        # 下载文件
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # 保存文件
        with open(output_path, 'wb') as f:
            f.write(response.content)

        size = len(response.content)
        return {'success': True, 'skipped': False, 'size': size, 'error': None}

    except Exception as e:
        return {'success': False, 'skipped': False, 'size': 0, 'error': str(e)}

def download_single_file_worker(args):
    """
    单个文件下载的工作函数，供ThreadPoolExecutor调用。
    
    Args:
        args (tuple): 包含下载参数的元组 (repo, commit_id, rel_path, output_dir)
    
    Returns:
        dict: 包含原始参数和下载结果的字典
    """
    repo, commit_id, rel_path, output_dir = args
    result = download_file_from_github(repo, commit_id, rel_path, output_dir)
    result.update({'repo': repo, 'rel_path': rel_path})
    return result

# ==============================================================================
# 2. 批处理函数 (多进程中运行的任务)
# ==============================================================================

def process_download_batch(tasks_batch, threads_per_process, process_id):
    """
    由单个进程执行的函数，该进程会创建一个线程池来处理一批下载任务。

    Args:
        tasks_batch (list): 分配给这个进程的下载任务列表
        threads_per_process (int): 每个进程内部使用的线程数
        process_id (int): 当前进程的ID

    Returns:
        dict: 该批次的处理结果统计
    """
    successful_downloads = 0
    failed_downloads = 0
    skipped_downloads = 0
    total_size = 0

    desc = f"[进程 {process_id:02d}]"
    with tqdm(total=len(tasks_batch), desc=desc, unit="文件", position=process_id) as pbar:
        with ThreadPoolExecutor(max_workers=threads_per_process) as executor:
            future_to_task = {executor.submit(download_single_file_worker, task): task for task in tasks_batch}

            for future in as_completed(future_to_task):
                result = future.result()

                if result['success']:
                    successful_downloads += 1
                    total_size += result['size']
                    if result['skipped']:
                        skipped_downloads += 1
                else:
                    failed_downloads += 1

                pbar.update(1)
                pbar.set_postfix({
                    '成功': successful_downloads - skipped_downloads,
                    '失败': failed_downloads,
                    '跳过': skipped_downloads,
                    '线程': threads_per_process
                })

    return {
        'successful_downloads': successful_downloads,
        'failed_downloads': failed_downloads,
        'skipped_downloads': skipped_downloads,
        'total_size': total_size,
        'process_id': process_id
    }

# ==============================================================================
# 3. Parquet文件处理主函数 (数据切片和进程分发)
# ==============================================================================

def process_single_parquet_file(file_path, base_output_dir, threads_per_process, max_processes):
    """
    处理单个parquet文件，进行数据切片并分配给多个进程。

    Args:
        file_path (str): 待处理的parquet文件路径
        base_output_dir (str): 基础输出目录
        threads_per_process (int): 每个进程的最大线程数
        max_processes (int): 最大进程数

    Returns:
        dict: 处理结果
    """
    try:
        parquet_filename = os.path.splitext(os.path.basename(file_path))[0]
        current_output_dir = os.path.join(base_output_dir, parquet_filename)
        Path(current_output_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"开始处理 Parquet 文件: {os.path.basename(file_path)}")
        print(f"文件将保存到: {current_output_dir}")
        print(f"使用 {max_processes} 个进程, 每个进程 {threads_per_process} 个线程进行处理。")
        print(f"{'-'*80}")

        df = pd.read_parquet(file_path)
        total_records = len(df)
        if total_records == 0:
            print("文件为空，跳过。")
            return {'success': True, 'total_files': 0, 'successful_downloads': 0, 'failed_downloads': 0, 'total_size': 0}

        # 准备所有下载任务
        all_download_tasks = [
            (row['repo'], row['commit_id'], row['rel_path'], current_output_dir)
            for _, row in df.iterrows()
        ]

        # 将下载任务分片，分配给不同进程
        tasks_per_process = math.ceil(total_records / max_processes)
        process_task_batches = []
        for i in range(max_processes):
            start_index = i * tasks_per_process
            end_index = start_index + tasks_per_process
            batch = all_download_tasks[start_index:end_index]
            if batch:
                process_task_batches.append((batch, threads_per_process, i + 1))

        # 使用多进程处理下载任务
        total_successful = 0
        total_failed = 0
        total_skipped = 0
        total_size = 0

        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            future_to_batch = {executor.submit(process_download_batch, *args): args for args in process_task_batches}

            for future in as_completed(future_to_batch):
                result = future.result()
                total_successful += result['successful_downloads']
                total_failed += result['failed_downloads']
                total_skipped += result['skipped_downloads']
                total_size += result['total_size']

        print(f"\n{'-'*80}")
        print(f"文件 {os.path.basename(file_path)} 处理完成:")
        print(f"  总记录数: {total_records:,}")
        print(f"  成功下载 (含跳过): {total_successful:,}")
        print(f"  其中新下载: {total_successful - total_skipped:,}")
        print(f"  其中已存在跳过: {total_skipped:,}")
        print(f"  下载失败: {total_failed:,}")
        print(f"  总大小: {total_size/1024/1024:.2f} MB")
        print(f"{'='*80}")

        return {
            'success': total_failed == 0,
            'total_files': total_records,
            'successful_downloads': total_successful,
            'failed_downloads': total_failed,
            'total_size': total_size
        }

    except Exception as e:
        print(f"\n处理文件 {os.path.basename(file_path)} 时发生严重错误: {str(e)}")
        return {'success': False, 'error': str(e), 'total_files': 0, 'successful_downloads': 0, 'failed_downloads': 0, 'total_size': 0}

# ==============================================================================
# 4. 顶层控制函数 (管理文件组和断点续下)
# ==============================================================================

def get_completed_files_path(output_dir, group_id):
    """获取记录已完成Parquet文件列表的路径"""
    return os.path.join(output_dir, f"completed_files_group_{group_id:04d}.txt")

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

def download_group_files(input_dir, output_dir, group_id, threads_per_process, max_processes):
    """
    下载指定文件组（group_id）的所有文件。
    该函数逐个处理Parquet文件，并将每个文件交给多进程系统处理。
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_parquet_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not all_parquet_files:
        print("错误: 输入目录中未找到任何 .parquet 文件。")
        return

    files_per_group = 50
    total_groups = math.ceil(len(all_parquet_files) / files_per_group)

    if group_id >= total_groups:
        print(f"错误: 块ID {group_id} 超出范围，总共有 {total_groups} 个块 (0-{total_groups-1})。")
        return

    start_index = group_id * files_per_group
    end_index = min(start_index + files_per_group, len(all_parquet_files))
    group_files = all_parquet_files[start_index:end_index]

    print(f"总共找到 {len(all_parquet_files)} 个 Parquet 文件, 分为 {total_groups} 个块。")
    print(f"当前任务: 下载块 {group_id} (文件索引 {start_index} 到 {end_index-1})")

    completed_files_path = get_completed_files_path(output_dir, group_id)
    completed_files = load_completed_files(completed_files_path)
    print(f"已加载 {len(completed_files)} 个已完成的 Parquet 文件记录。")

    remaining_files = [f for f in group_files if os.path.basename(f) not in completed_files]
    
    if not remaining_files:
        print(f"\n块 {group_id} 中的所有 Parquet 文件均已处理完毕！")
        return

    print(f"块 {group_id} 中还有 {len(remaining_files)} 个 Parquet 文件待处理。")
    start_time = time.time()

    for i, file_path in enumerate(remaining_files):
        parquet_filename = os.path.basename(file_path)
        print(f"\n>>> 开始处理块内文件 {i+1}/{len(remaining_files)}: {parquet_filename} <<<")
        
        result = process_single_parquet_file(
            file_path,
            output_dir,
            threads_per_process,
            max_processes
        )

        if result.get('success', False):
            save_completed_file(completed_files_path, parquet_filename)
            print(f"✓ Parquet 文件 {parquet_filename} 已成功处理并标记为完成。")
        else:
            print(f"✗ Parquet 文件 {parquet_filename} 处理失败，错误: {result.get('error', '未知错误')}")
            print("脚本将继续处理下一个文件。")

    end_time = time.time()
    print(f"\n{'*'*80}")
    print(f"块 {group_id} 全部任务处理完成! 总耗时: {(end_time - start_time)/60:.2f} 分钟")
    print(f"*'*80'")

# ==============================================================================
# 5. 主程序入口
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='从Parquet文件中下载GitHub文件，使用多进程对单个Parquet文件进行数据分片下载。',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input_dir', default='/madehua/data/Nemotron-Pretraining-Code-v1/Nemotron-Code-Metadata',
                        help='包含Parquet文件的输入目录路径')
    parser.add_argument('--output_dir', default='/mnt/oprover-data',
                        help='保存下载文件的输出目录路径')
    parser.add_argument('--threads_per_process', type=int, default=16,
                        help='每个下载进程内部使用的线程数 (默认: 16)')
    parser.add_argument('--processes', type=int, default=mp.cpu_count(),
                        help=f'用于处理单个Parquet文件的最大进程数 (默认: 系统CPU核心数, 即 {mp.cpu_count()})')
    parser.add_argument('--group_id', type=int, required=True,
                        help='要下载的文件块ID (从0开始，每个块默认50个Parquet文件)')
    
    args = parser.parse_args()
    
    print("下载任务启动参数:")
    print(f"  - 输入目录: {args.input_dir}")
    print(f"  - 输出目录: {args.output_dir}")
    print(f"  - 使用进程数: {args.processes}")
    print(f"  - 每进程线程数: {args.threads_per_process}")
    print(f"  - 目标块ID: {args.group_id}")
    print(f"="*60)
    
    download_group_files(args.input_dir, args.output_dir, args.group_id, args.threads_per_process, args.processes)

if __name__ == "__main__":
    # 在多进程代码中，建议将主逻辑放在 if __name__ == "__main__": 块内
    main()