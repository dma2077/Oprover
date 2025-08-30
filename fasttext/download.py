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

def download_file_from_github(repo, commit_id, rel_path, output_dir):
    """
    从GitHub下载单个文件
    
    Args:
        repo: 仓库名称
        commit_id: 提交ID
        rel_path: 相对路径
        output_dir: 输出目录
    
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
        
        # 如果文件已存在，跳过下载
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            return {
                'success': True,
                'filename': filename,
                'size': size,
                'error': None
            }
        
        # 下载文件
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # 保存文件
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        size = len(response.content)
        
        return {
            'success': True,
            'filename': filename,
            'size': size,
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'filename': None,
            'size': 0,
            'error': str(e)
        }

def download_single_file_worker(args):
    """
    单个文件下载的工作函数，用于多线程调用
    
    Args:
        args: 包含下载参数的元组 (repo, commit_id, rel_path, output_dir, file_index)
    
    Returns:
        dict: 下载结果
    """
    repo, commit_id, rel_path, output_dir, file_index = args
    
    try:
        # 下载文件
        result = download_file_from_github(repo, commit_id, rel_path, output_dir)
        
        # 添加文件索引信息
        result['file_index'] = file_index
        result['repo'] = repo
        result['rel_path'] = rel_path
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'filename': None,
            'size': 0,
            'error': str(e),
            'file_index': file_index,
            'repo': repo,
            'rel_path': rel_path
        }

def get_checkpoint_path(output_dir, group_id):
    """
    获取检查点文件路径
    
    Args:
        output_dir: 输出目录
        group_id: 块ID
    
    Returns:
        str: 检查点文件路径
    """
    return os.path.join(output_dir, f"checkpoint_group_{group_id:04d}.json")

def load_checkpoint(checkpoint_path):
    """
    加载检查点信息
    
    Args:
        checkpoint_path: 检查点文件路径
    
    Returns:
        dict: 检查点信息，如果不存在则返回None
    """
    try:
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"加载检查点失败: {e}")
    return None

def save_checkpoint(checkpoint_path, checkpoint_data):
    """
    保存检查点信息
    
    Args:
        checkpoint_path: 检查点文件路径
        checkpoint_data: 检查点数据
    """
    try:
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存检查点失败: {e}")

def get_completed_files_path(output_dir, group_id):
    """
    获取已完成文件列表的路径
    
    Args:
        output_dir: 输出目录
        group_id: 块ID
    
    Returns:
        str: 已完成文件列表路径
    """
    return os.path.join(output_dir, f"completed_files_group_{group_id:04d}.txt")

def load_completed_files(completed_files_path):
    """
    加载已完成的parquet文件列表
    
    Args:
        completed_files_path: 已完成文件列表路径
    
    Returns:
        set: 已完成的文件名集合
    """
    completed_files = set()
    try:
        if os.path.exists(completed_files_path):
            with open(completed_files_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        completed_files.add(line)
            print(f"加载已完成文件列表: {len(completed_files)} 个文件")
        else:
            print("未找到已完成文件列表，将从头开始下载")
    except Exception as e:
        print(f"加载已完成文件列表失败: {e}")
    
    return completed_files

def save_completed_file(completed_files_path, parquet_filename):
    """
    将已完成的parquet文件添加到完成列表
    
    Args:
        completed_files_path: 已完成文件列表路径
        parquet_filename: 已完成的parquet文件名
    """
    try:
        with open(completed_files_path, 'a', encoding='utf-8') as f:
            f.write(f"{parquet_filename}\n")
    except Exception as e:
        print(f"保存已完成文件记录失败: {e}")

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

def process_download_batch(args):
    """
    多进程版本的单个批次下载处理函数
    
    Args:
        args: 包含下载参数的元组 (tasks, max_workers, process_id, checkpoint_key)
    
    Returns:
        dict: 处理结果
    """
    tasks, max_workers, process_id, checkpoint_key = args
    
    try:
        print(f"[进程 {os.getpid()}] 批次 {process_id} 开始处理 {len(tasks)} 个下载任务")
        
        successful_downloads = 0
        failed_downloads = 0
        total_size = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(download_single_file_worker, task): task for task in tasks}
            
            with tqdm(total=len(tasks), desc=f"[进程 {os.getpid()}] 批次 {process_id} 下载进度", unit="文件") as pbar:
                for future in as_completed(future_to_task):
                    result = future.result()
                    
                    if result['success'] and result['filename']:
                        successful_downloads += 1
                        total_size += result['size']
                    else:
                        failed_downloads += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Repo': result['repo'][:20] + '...' if len(result['repo']) > 20 else result['repo'],
                        'File': result['rel_path'][:30] + '...' if len(result['rel_path']) > 30 else result['rel_path'],
                        '成功': successful_downloads,
                        '失败': failed_downloads,
                        '线程': max_workers
                    })
        
        print(f"[进程 {os.getpid()}] 批次 {process_id} 完成: 成功 {successful_downloads}, 失败 {failed_downloads}")
        
        return {
            'successful_downloads': successful_downloads,
            'failed_downloads': failed_downloads,
            'total_size': total_size,
            'process_id': process_id
        }
        
    except Exception as e:
        print(f"[进程 {os.getpid()}] 批次 {process_id} 处理失败: {str(e)}")
        return {
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_size': 0,
            'process_id': process_id
        }

def process_parquet_file_with_data_slicing(file_path, output_dir, max_workers, max_processes, checkpoint_data, group_id, file_index):
    """
    处理单个parquet文件，进行数据切片并分配给多个进程
    
    Args:
        file_path: 待处理的parquet文件路径
        output_dir: 输出目录
        max_workers: 每个进程的最大线程数
        max_processes: 最大进程数
        checkpoint_data: 检查点数据
        group_id: 块ID
        file_index: 当前parquet文件在块中的索引
    
    Returns:
        dict: 处理结果
    """
    try:
        print(f"开始处理文件: {os.path.basename(file_path)}")
        
        # 读取parquet文件
        df = pd.read_parquet(file_path)
        total_files = len(df)
        
        print(f"文件包含 {total_files:,} 条记录")
        print(f"使用 {max_processes} 个进程，每个进程 {max_workers} 个线程")
        
        # 检查点键
        checkpoint_key = f"group_{group_id:04d}_file_{file_index:04d}"
        
        # 从检查点恢复状态
        if checkpoint_data and checkpoint_key in checkpoint_data:
            resume_info = checkpoint_data[checkpoint_key]
            start_index = resume_info.get('completed_files', 0)
            print(f"从检查点恢复，已完成 {start_index:,} 个文件")
        else:
            start_index = 0
            print(f"开始新的下载任务")
        
        # 创建以parquet文件名命名的目录
        parquet_filename = os.path.splitext(os.path.basename(file_path))[0]
        current_dir = os.path.join(output_dir, parquet_filename)
        
        # 创建目录
        Path(current_dir).mkdir(parents=True, exist_ok=True)
        print(f"文件将保存到目录: {current_dir}")
        
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
        
        if not download_tasks:
            print("没有需要下载的文件")
            return {
                'filename': os.path.basename(file_path),
                'total_files': total_files,
                'successful_downloads': successful_downloads,
                'failed_downloads': failed_downloads,
                'total_size': total_size,
                'success': True,
                'error': None
            }
        
        # 将下载任务分配给多个进程（真正的数据切片）
        if max_processes == 1:
            # 单进程模式：使用多线程
            print(f"单进程模式：使用 {max_workers} 个线程处理所有任务")
            
            # 使用多线程下载
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有下载任务
                future_to_task = {executor.submit(download_single_file_worker, task): task for task in download_tasks}
                
                # 使用tqdm显示下载进度
                with tqdm(total=len(download_tasks), desc=f"下载进度", unit="文件") as pbar:
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
        else:
            # 多进程模式：数据切片
            tasks_per_process = math.ceil(len(download_tasks) / max_processes)
            
            print(f"将 {len(download_tasks)} 个下载任务分配给 {max_processes} 个进程")
            print(f"每个进程处理约 {tasks_per_process} 个任务")
            
            # 使用多进程处理下载任务
            with ProcessPoolExecutor(max_workers=max_processes) as executor:
                # 准备进程参数
                process_args = []
                for i in range(max_processes):
                    start_task = i * tasks_per_process
                    end_task = min(start_task + tasks_per_process, len(download_tasks))
                    
                    if start_task < len(download_tasks):
                        process_tasks = download_tasks[start_task:end_task]
                        process_args.append((
                            process_tasks,
                            max_workers,
                            i,  # 进程ID
                            checkpoint_key
                        ))
                
                # 提交所有进程任务
                future_to_process = {executor.submit(process_download_batch, args): args for args in process_args}
                
                # 收集结果
                for future in as_completed(future_to_process):
                    result = future.result()
                    
                    # 更新统计信息
                    successful_downloads += result['successful_downloads']
                    failed_downloads += result['failed_downloads']
                    total_size += result['total_size']
                    
                    print(f"进程 {result['process_id']} 完成: 成功 {result['successful_downloads']}, 失败 {result['failed_downloads']}")
        
        print(f"\n文件 {os.path.basename(file_path)} 处理完成:")
        print(f"  成功下载: {successful_downloads:,}")
        print(f"  下载失败: {failed_downloads:,}")
        print(f"  总大小: {total_size/1024/1024:.2f} MB")
        print(f"  使用进程数: {max_processes}")
        print(f"  每个进程线程数: {max_workers}")
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
        print(f"\n处理文件 {os.path.basename(file_path)} 时出错: {str(e)}")
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
    实现真正的数据切片：将一个parquet文件中的数据切片分配给多个进程处理
    
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
    print(f"策略: 将parquet文件中的数据切片分配给多个进程处理")
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
    
    # 使用多进程处理文件
    start_time = time.time()
    total_successful = 0
    total_failed = 0
    total_size = 0
    
    # 逐个处理parquet文件，每个文件内部进行数据切片
    for file_index, file_path in enumerate(remaining_files):
        parquet_filename = os.path.basename(file_path)
        print(f"\n{'='*60}")
        print(f"开始处理文件 {file_index + 1}/{len(remaining_files)}: {parquet_filename}")
        print(f"{'='*60}")
        
        # 处理单个parquet文件，进行数据切片
        result = process_parquet_file_with_data_slicing(
            file_path, 
            output_dir, 
            max_workers, 
            max_processes,
            checkpoint_data,
            group_id,
            file_index
        )
        
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
    parser.add_argument('--processes', type=int, default=64,
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