#!/usr/bin/env python3
"""
计算DeepSeek-Prover-V2-7B模型的pass@k指标
pass@k的定义：对于每个query，前k个response中至少有一个成功的概率
"""

import json
import os
import glob
from collections import defaultdict
from typing import Dict, List, Tuple

def parse_id(id_str: str) -> Tuple[str, int]:
    """解析ID格式，例如 '151413_23' -> ('151413', 23)"""
    parts = id_str.split('_')
    if len(parts) != 2:
        raise ValueError(f"Invalid ID format: {id_str}")
    query_id = parts[0]
    response_idx = int(parts[1])  # 这里是从0开始的index
    return query_id, response_idx

def load_verified_data(file_path: str) -> Dict[str, List[Tuple[int, bool]]]:
    """
    从指定的结果文件加载数据
    返回格式: {query_id: [(response_idx, success), ...]}
    """
    query_responses = defaultdict(list)
    
    print(f"Processing file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    id_str = data['id']
                    success = data['success']
                    
                    query_id, response_idx = parse_id(id_str)
                    query_responses[query_id].append((response_idx, success))
                    
                    line_count += 1
                    if line_count % 10000 == 0:
                        print(f"  Processed {line_count} lines")
                        
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"Error processing line: {e}")
                    continue
                    
            print(f"  Finished processing: {line_count} lines")
            
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return {}
    
    # 对每个query的responses按response_idx排序
    for query_id in query_responses:
        query_responses[query_id].sort(key=lambda x: x[0])
    
    return dict(query_responses)

def calculate_pass_at_k(query_responses: Dict[str, List[Tuple[int, bool]]], k_values: List[int]) -> Dict[int, Tuple[int, int, float]]:
    """
    计算pass@k指标
    返回格式: {k: (total_queries, successful_queries, pass_rate)}
    """
    results = {}
    
    for k in k_values:
        total_queries = 0
        successful_queries = 0
        
        for query_id, responses in query_responses.items():
            # 只考虑前k个response
            responses_k = responses[:k]
            
            if len(responses_k) == 0:
                # 如果这个query没有任何response，跳过
                continue
                
            total_queries += 1
            
            # 检查前k个response中是否至少有一个成功
            has_success = any(success for _, success in responses_k)
            if has_success:
                successful_queries += 1
        
        pass_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0.0
        results[k] = (total_queries, successful_queries, pass_rate)
    
    return results

def calculate_improvements(results: Dict[int, Tuple[int, int, float]]) -> Dict[str, float]:
    """计算从k-1到k的平均每次尝试提升"""
    improvements = {}
    k_values = sorted(results.keys())
    
    # 定义提升对应关系
    improvement_pairs = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32)]
    
    for k_prev, k_curr in improvement_pairs:
        if k_prev in results and k_curr in results:
            _, _, pass_rate_prev = results[k_prev]
            _, _, pass_rate_curr = results[k_curr]
            total_improvement = pass_rate_curr - pass_rate_prev
            # 计算平均每次尝试的提升
            attempts_added = k_curr - k_prev
            avg_improvement_per_attempt = total_improvement / attempts_added
            improvements[f"{k_prev}→{k_curr}"] = avg_improvement_per_attempt
    
    return improvements

def print_results(results: Dict[int, Tuple[int, int, float]]):
    """打印结果"""
    print("\n" + "="*80)
    print("Pass@k Results and Improvements")
    print("="*80)
    print(f"{'k':<4} {'Total Queries':<15} {'Successful':<12} {'Pass@k':<10} {'Avg Per Try':<15}")
    print("-"*80)
    
    # 计算提升
    improvements = calculate_improvements(results)
    
    for k in sorted(results.keys()):
        total, successful, pass_rate = results[k]
        
        # 查找对应的提升值
        improvement_str = ""
        for imp_key, imp_value in improvements.items():
            if imp_key.endswith(f"→{k}"):
                improvement_str = f"+{imp_value:.2f}"
                break
        
        print(f"{k:<4} {total:<15} {successful:<12} {pass_rate:<10.2f} {improvement_str:<15}")
    
    print("="*80)

def analyze_query_distribution(query_responses: Dict[str, List[Tuple[int, bool]]]):
    """分析query的response数量分布"""
    print("\n" + "="*60)
    print("Query Response Distribution Analysis")
    print("="*60)
    
    response_counts = defaultdict(int)
    success_counts = defaultdict(int)
    
    for query_id, responses in query_responses.items():
        num_responses = len(responses)
        response_counts[num_responses] += 1
        
        # 统计成功的response数量
        num_successes = sum(1 for _, success in responses if success)
        success_counts[num_successes] += 1
    
    print("Response count distribution:")
    for count in sorted(response_counts.keys()):
        print(f"  {count} responses: {response_counts[count]} queries")
    
    print("\nSuccess count distribution:")
    for count in sorted(success_counts.keys()):
        print(f"  {count} successes: {success_counts[count]} queries")
    
    print(f"\nTotal unique queries: {len(query_responses)}")
    
    # 统计至少有一个成功的query数量
    queries_with_success = sum(1 for responses in query_responses.values() 
                             if any(success for _, success in responses))
    print(f"Queries with at least one success: {queries_with_success}")
    print(f"Overall success rate: {queries_with_success / len(query_responses) * 100:.2f}")

def main():
    import sys
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("Usage: python calculate_pass_at_k.py <result_file_path>")
        print("Example: python calculate_pass_at_k.py /path/to/your/result_file.jsonl")
        sys.exit(1)
    
    # 从命令行参数获取结果文件路径
    result_file = sys.argv[1]
    
    # 检查文件是否存在
    if not os.path.exists(result_file):
        print(f"Error: File not found: {result_file}")
        sys.exit(1)
    
    # 要计算的k值
    k_values = [1, 2, 4, 8, 16, 32]
    
    print("Loading result data...")
    query_responses = load_verified_data(result_file)
    print(f"\nLoaded data for {len(query_responses)} unique queries")
    
    # 分析数据分布
    analyze_query_distribution(query_responses)
    
    # 计算pass@k
    print("\nCalculating pass@k metrics...")
    results = calculate_pass_at_k(query_responses, k_values)
    
    # 打印结果
    print_results(results)
    
    # 计算提升数据
    improvements = calculate_improvements(results)
    
    # 保存结果到文件，基于输入文件名生成输出文件名
    input_basename = os.path.splitext(os.path.basename(result_file))[0]
    output_file = f"/data/code/Oprover/post_process/{input_basename}_pass_at_k_results.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'input_file': result_file,
            'results': {str(k): {'total_queries': total, 'successful_queries': successful, 'pass_rate': round(pass_rate, 2)}
                       for k, (total, successful, pass_rate) in results.items()},
            'improvements': {transition: round(improvement, 2) for transition, improvement in improvements.items()},
            'summary': {
                'total_unique_queries': len(query_responses),
                'k_values': k_values,
                'average_improvement': round(sum(improvements.values()) / len(improvements), 2) if improvements else 0
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
