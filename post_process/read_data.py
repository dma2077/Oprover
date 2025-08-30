import json
import os
from tqdm import tqdm

def analyze_response_n_average(file_path):
    """
    读取一个 .jsonl 文件，统计其中 'response_n' 字段的平均元素数量。

    Args:
        file_path (str): .jsonl 文件的路径。
    """
    # --- 1. 检查文件是否存在 ---
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 -> '{file_path}'")
        return

    total_elements = 0  # 存储所有 'response_n' 列表中的元素总数
    record_count = 0    # 存储包含 'response_n' 列表的记录总数
    line_number = 0     # 当前行号，用于错误提示

    print(f"开始分析文件: {os.path.basename(file_path)}")

    try:
        # --- 2. 打开文件并逐行处理 ---
        # 使用 with 语句确保文件被正确关闭
        with open(file_path, 'r', encoding='utf-8') as f:
            # 为了给大文件处理提供进度条，我们先快速统计总行数
            # 这可能会花费一些时间，但对于用户体验是值得的
            print("正在计算文件总行数以便显示进度...")
            num_lines = sum(1 for line in open(file_path, 'r', encoding='utf-8'))
            f.seek(0) # 重置文件指针到开头

            # 使用 tqdm 显示处理进度
            progress_bar = tqdm(f, total=num_lines, desc="正在处理行", unit="行")

            for line in progress_bar:
                line_number += 1
                line = line.strip()

                # 跳过空行
                if not line:
                    continue

                try:
                    # --- 3. 解析 JSON 并提取数据 ---
                    data = json.loads(line)
                    
                    # 获取 'response_n' 字段的值
                    response_n_value = data.get('response_n')

                    # 检查 'response_n' 是否是一个列表
                    if isinstance(response_n_value, list):
                        total_elements += len(response_n_value)
                        record_count += 1

                except json.JSONDecodeError:
                    print(f"\n警告: 第 {line_number} 行不是有效的JSON格式，已跳过。")
                    continue
    
    except FileNotFoundError:
        print(f"错误: 文件未找到 -> '{file_path}'")
        return
    except Exception as e:
        print(f"\n处理文件时发生未知错误: {e}")
        return

    # --- 4. 计算并打印最终结果 ---
    print("\n--- 分析结果 ---")
    if record_count > 0:
        average_elements = total_elements / record_count
        print(f"在 {record_count:,} 条包含 'response_n' 列表的记录中:")
        print(f"  - 元素总数: {total_elements:,}")
        print(f"  - 平均元素数量: {average_elements:.2f}")
    else:
        print("文件中没有找到任何包含 'response_n' 字段的有效记录。")

if __name__ == "__main__":
    # --- 请在这里设置您的文件路径 ---
    jsonl_file_path = "/data/code/Oprover/Goedel-Prover-V2-8B_results/Goedel-Prover-V2-8B_FineLeanCorpus_20_filter/lean_statement_part_01_proof_cot-bon.jsonl"
    
    # 运行分析函数
    analyze_response_n_average(jsonl_file_path)