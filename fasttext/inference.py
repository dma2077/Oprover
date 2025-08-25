#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import fasttext
from pathlib import Path
from tqdm import tqdm
import argparse
import time

def load_fasttext_model(model_path):
    """加载FastText模型"""
    print(f"正在加载FastText模型: {model_path}")
    try:
        model = fasttext.load_model(model_path)
        print(f"模型加载成功")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise

def predict_label(model, text):
    """使用FastText模型预测标签"""
    # FastText预测返回 (标签列表, 概率列表)
    # 标签格式通常是 '__label__0' 或 '__label__1'
    labels, probs = model.predict(text.replace('\n', ' '))
    
    # 提取数字标签
    label = labels[0]
    if '__label__' in label:
        label_num = int(label.replace('__label__', ''))
    else:
        # 如果标签不是标准格式，尝试直接转换
        label_num = int(label)
    
    return label_num, probs[0]

def process_jsonl_file(input_file, output_file, model):
    """处理单个JSONL文件，只保留label=1的数据并添加score字段"""
    retained_lines = []
    total_lines = 0
    retained_count = 0
    
    # 读取文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_valid_lines = len([l for l in lines if l.strip()])
    
    # 使用tqdm显示单个文件的处理进度
    for line in tqdm(lines, desc=f"处理 {os.path.basename(input_file)}", leave=False):
        line = line.strip()
        if not line:
            continue
        
        total_lines += 1
            
        try:
            # 解析JSON
            data = json.loads(line)
            
            # 获取text字段
            if 'text' not in data:
                print(f"警告: 缺少text字段，跳过该行")
                continue
            
            text = data['text']
            
            # 预测标签
            label, confidence = predict_label(model, text)
            
            # 只保留label=1的数据
            if label == 1:
                # 添加score字段
                data['score'] = round(confidence, 4)  # 保留4位小数
                retained_lines.append(json.dumps(data, ensure_ascii=False))
                retained_count += 1
            
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            continue
        except Exception as e:
            print(f"处理错误: {e}")
            continue
    
    # 写入输出文件（只包含label=1的数据，包含score字段）
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in retained_lines:
            f.write(line + '\n')
    
    # 计算留存率
    retention_rate = (retained_count / total_lines * 100) if total_lines > 0 else 0
    
    return total_lines, retained_count, retention_rate

def process_directory(input_dir, output_dir, model_path):
    """处理整个目录"""
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    model = load_fasttext_model(model_path)
    
    # 获取所有jsonl文件
    jsonl_files = glob.glob(os.path.join(input_dir, '*.jsonl'))
    
    if not jsonl_files:
        print(f"在 {input_dir} 中没有找到JSONL文件")
        return
    
    print(f"找到 {len(jsonl_files)} 个JSONL文件")
    print("="*50)
    
    # 统计信息
    total_processed = 0
    total_retained = 0
    successful_files = 0
    failed_files = []
    file_stats = []
    
    # 处理每个文件
    start_time = time.time()
    
    for idx, jsonl_file in enumerate(tqdm(jsonl_files, desc="总体进度"), 1):
        try:
            # 构建输出文件路径
            filename = os.path.basename(jsonl_file)
            output_file = os.path.join(output_dir, filename)
            
            # 处理文件
            total_lines, retained_count, retention_rate = process_jsonl_file(
                jsonl_file, output_file, model
            )
            
            # 更新累计统计
            total_processed += total_lines
            total_retained += retained_count
            successful_files += 1
            
            # 计算当前累计留存率
            cumulative_retention_rate = (total_retained / total_processed * 100) if total_processed > 0 else 0
            
            # 记录统计信息
            file_stats.append({
                'filename': filename,
                'total': total_lines,
                'retained': retained_count,
                'retention_rate': retention_rate
            })
            
            # 显示单个文件的处理结果
            print(f"\n[{idx}/{len(jsonl_files)}] ✓ {filename}")
            print(f"  本文件: {total_lines} 行 -> {retained_count} 行 (留存率: {retention_rate:.2f}%)")
            print(f"  累计统计: 总处理 {total_processed} 行, 总留存 {total_retained} 行 (总留存率: {cumulative_retention_rate:.2f}%)")
            
        except Exception as e:
            print(f"\n✗ 失败: {filename} - {e}")
            failed_files.append(filename)
    
    # 打印最终统计信息
    elapsed_time = time.time() - start_time
    overall_retention_rate = (total_retained / total_processed * 100) if total_processed > 0 else 0
    
    print("\n" + "="*50)
    print("处理完成！")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"成功处理文件: {successful_files}/{len(jsonl_files)}")
    print("\n最终统计:")
    print(f"  原始总行数: {total_processed}")
    print(f"  留存总行数: {total_retained}")
    print(f"  总体留存率: {overall_retention_rate:.2f}%")
    
    # 显示每个文件的详细统计
    if file_stats:
        print("\n各文件留存率详情:")
        print("-" * 60)
        print(f"{'文件名':<30} {'原始':<10} {'留存':<10} {'留存率':<10}")
        print("-" * 60)
        for stat in file_stats:
            print(f"{stat['filename']:<30} {stat['total']:<10} {stat['retained']:<10} {stat['retention_rate']:<.2f}%")
    
    if failed_files:
        print(f"\n失败文件列表:")
        for f in failed_files:
            print(f"  - {f}")
    
    print(f"\n输出目录: {output_dir}")
    
    # 保存统计报告
    report_file = os.path.join(output_dir, 'filter_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"FastText过滤报告\n")
        f.write(f"="*50 + "\n")
        f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {elapsed_time:.2f} 秒\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"输入目录: {input_dir}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"\n总体统计:\n")
        f.write(f"  原始总行数: {total_processed}\n")
        f.write(f"  留存总行数: {total_retained}\n")
        f.write(f"  总体留存率: {overall_retention_rate:.2f}%\n")
        f.write(f"\n各文件统计:\n")
        for stat in file_stats:
            f.write(f"  {stat['filename']}: {stat['total']} -> {stat['retained']} ({stat['retention_rate']:.2f}%)\n")
    
    print(f"统计报告已保存至: {report_file}")
    
    # 保存JSON格式的统计报告（方便程序化处理）
    json_report_file = os.path.join(output_dir, 'filter_report.json')
    json_report = {
        'process_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_seconds': round(elapsed_time, 2),
        'model_path': model_path,
        'input_dir': input_dir,
        'output_dir': output_dir,
        'summary': {
            'total_files': len(jsonl_files),
            'successful_files': successful_files,
            'failed_files': len(failed_files),
            'total_lines': total_processed,
            'retained_lines': total_retained,
            'overall_retention_rate': round(overall_retention_rate, 2)
        },
        'file_details': file_stats,
        'failed_files': failed_files
    }
    
    with open(json_report_file, 'w', encoding='utf-8') as f:
        json.dump(json_report, f, ensure_ascii=False, indent=2)
    
    print(f"JSON报告已保存至: {json_report_file}")

def main():
    parser = argparse.ArgumentParser(description='使用FastText模型过滤JSONL文件（只保留正例并添加分数）')
    parser.add_argument('input_dir', help='输入目录路径（包含JSONL文件）')
    parser.add_argument('output_dir', help='输出目录路径')
    parser.add_argument('model_path', help='FastText模型路径（model.bin）')
    parser.add_argument('--min-confidence', type=float, default=0.0, 
                       help='最小置信度阈值（可选，默认0.0，即不过滤）')
    
    args = parser.parse_args()
    
    # 验证路径
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return
    
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    # 开始处理
    process_directory(args.input_dir, args.output_dir, args.model_path)

if __name__ == "__main__":
    main()

# 使用示例:
# python fasttext_filter.py /path/to/input/dir /path/to/output/dir model.bin
# 
# 带置信度阈值:
# python fasttext_filter.py /path/to/input/dir /path/to/output/dir model.bin --min-confidence 0.8
