#!/usr/bin/env python3
"""
从所有parquet文件中下载GitHub文件
使用多线程和进度条，按1000个文件分组到不同目录
支持断点续下和分块下载
"""

import os
import pandas as pd
import requests
import time
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re
import glob
import math
from collections import defaultdict
import threading
import json
import pickle

# 线程锁，用于保护共享资源
lock = threading.Lock()

def sanitize_filename(filename):
    """
    清理文件名，移除或替换不合法的字符
    """
    # 替换不合法的文件名字符
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 移除多余的空格和点
    filename = re.sub(r'\s+', '_', filename)
    filename = re.sub(r'\.+', '.', filename)
    # 限制文件名长度
    if len(filename) > 200:
        filename = filename[:200]
    return filename

def get_file_extension(rel_path):
    """
    从相对路径中提取文件扩展名
    """
    _, ext = os.path.splitext(rel_path)
    return ext

def download_file_from_github(repo, commit_id, rel_path, output_dir):
    """
    从GitHub下载指定commit的文件
    
    Args:
        repo: 仓库名 (格式: owner/repo)
        commit_id: 提交ID
        rel_path: 相对路径
        output_dir: 输出目录
    
    Returns:
        dict: 下载结果
    """
    try:
        # 构建GitHub API URL
        api_url = f"https://raw.githubusercontent.com/{repo}/{commit_id}/{rel_path}"
        
        # 发送请求
        response = requests.get(api_url, timeout=30)
        
        if response.status_code == 200:
            # 获取文件扩展名
            file_ext = get_file_extension(rel_path)
            
            # 构建文件名: repo_commit_id_rel_path
            # 将repo中的/替换为_
            repo_clean = repo.replace('/', '_')
            
            # 将rel_path中的/替换为_
            rel_path_clean = rel_path.replace('/', '_')
            
            # 组合文件名
            filename = f"{repo_clean}_{commit_id}_{rel_path_clean}"
            
            # 清理文件名
            filename = sanitize_filename(filename)
            
            # 确保文件名有正确的扩展名
            if file_ext and not filename.endswith(file_ext):
                filename += file_ext
            
            # 构建输出路径
            output_path = os.path.join(output_dir, filename)
            
            # 保存文件
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            return {
                'success': True,
                'filename': filename,
                'size': len(response.content),
                'error': None
            }
        else:
            return {
                'success': False,  # 修复：HTTP错误应该返回False
                'filename': None,
                'size': 0,
                'error': f"HTTP {response.status_code}"
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

def process_single_parquet_file(file_path, output_base_dir, max_workers=16, 
                               checkpoint_data=None, group_id=None, file_index_in_group=None):
    """
    处理单个parquet文件，下载其中的所有文件，使用多线程，支持断点续下
    
    Args:
        file_path: parquet文件路径
        output_base_dir: 输出基础目录
        max_workers: 最大线程数
        checkpoint_data: 检查点数据
        group_id: 块ID
        file_index_in_group: 在块中的文件索引
    
    Returns:
        dict: 处理结果
    """
    try:
        print(f"\n开始处理文件: {os.path.basename(file_path)} (块 {group_id}, 文件 {file_index_in_group})")
        
        # 读取parquet文件
        df = pd.read_parquet(file_path)
        total_files = len(df)
        
        print(f"文件包含 {total_files:,} 条记录")
        print(f"使用 {max_workers} 个线程进行下载")
        
        # 检查点键
        checkpoint_key = f"group_{group_id:04d}_file_{file_index_in_group:04d}"
        
        # 从检查点恢复状态
        if checkpoint_data and checkpoint_key in checkpoint_data:
            resume_info = checkpoint_data[checkpoint_key]
            start_index = resume_info.get('completed_files', 0)
            print(f"从检查点恢复，已完成 {start_index:,} 个文件")
        else:
            start_index = 0
            print("开始新的下载任务")
        
        # 创建以parquet文件名命名的目录
        parquet_filename = os.path.splitext(os.path.basename(file_path))[0]  # 去掉.parquet扩展名
        current_dir = os.path.join(output_base_dir, parquet_filename)
        
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
        
        # 使用多线程下载
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有下载任务
            future_to_task = {executor.submit(download_single_file_worker, task): task for task in download_tasks}
            
            # 使用tqdm显示下载进度
            with tqdm(total=total_files, initial=start_index, desc=f"下载进度", unit="文件") as pbar:
                for future in as_completed(future_to_task):
                    result = future.result()
                    
                    # 更新统计信息
                    with lock:
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
        
        print(f"\n文件 {os.path.basename(file_path)} 处理完成:")
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

def download_group_files(input_dir, output_dir, group_id, max_workers=16):
    """
    下载指定块ID的parquet文件，支持parquet文件级别的断点续下
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        group_id: 块ID (从0开始)
        max_workers: 最大线程数
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有parquet文件
    parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    parquet_files.sort()  # 按文件名排序
    
    if not parquet_files:
        print("未找到parquet文件")
        return
    
    total_files = len(parquet_files)
    files_per_group = 50  # 每个块50个文件
    
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
    print(f"线程数: {max_workers}")
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
    
    # 处理当前块的所有文件
    start_time = time.time()
    total_successful = 0
    total_failed = 0
    total_size = 0
    
    for i, file_path in enumerate(remaining_files):
        file_index_in_group = start_file_index + i
        parquet_filename = os.path.basename(file_path)
        
        print(f"\n开始处理文件 {i+1}/{len(remaining_files)}: {parquet_filename}")
        
        # 处理单个文件
        result = process_single_parquet_file(
            file_path, 
            output_dir, 
            max_workers,
            checkpoint_data,
            group_id,
            i
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
    print(f"使用线程数: {max_workers}")
    print(f"输出目录: {output_dir}")
    print(f"检查点文件: {checkpoint_path}")
    print(f"已完成文件列表: {completed_files_path}")
    print(f"="*60)

def main():
    parser = argparse.ArgumentParser(description='从parquet文件中下载GitHub文件，支持分块下载和断点续下')
    parser.add_argument('--input_dir', default='/madehua/data/Nemotron-Pretraining-Code-v1/Nemotron-Code-Metadata',
                       help='输入目录路径')
    parser.add_argument('--output_dir', default='/mnt/oprover-data',
                       help='输出目录路径')
    parser.add_argument('--threads', type=int, default=256,
                       help='下载线程数 (默认: 128)')
    parser.add_argument('--group_id', type=int, required=True,
                       help='块ID (从0开始，每个块包含50个parquet文件)')
    
    args = parser.parse_args()
    
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"线程数: {args.threads}")
    print(f"块ID: {args.group_id}")
    print(f"="*60)
    
    # 下载指定块的文件
    download_group_files(args.input_dir, args.output_dir, args.group_id, args.threads)

if __name__ == "__main__":
    main()
