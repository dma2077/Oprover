#!/usr/bin/env python3
import json

def filter_and_convert_data():
    input_file = "/home/i-madehua/code/Oprover/data/results_new.jsonl"
    output_file = "/home/i-madehua/code/Oprover/data/results_new_filtered.jsonl"
    
    filtered_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                # 检查条件：success为true且difficulty_score>=5
                difficulty_score = data.get('difficulty_score')
                if data.get('success') == True and difficulty_score is not None and isinstance(difficulty_score, (int, float)) and difficulty_score >= 5:
                    # 转换格式
                    converted_data = {
                        "id": data.get('uuid'),
                        "lean_code": data.get('formal_statement'),
                        "difficulty": data.get('difficulty_score'),
                        "domain": None,
                        "source": "NuminaMath-LEAN",
                        "statement": None
                    }
                    filtered_data.append(converted_data)
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    # 保存过滤后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"处理完成！")
    print(f"原始数据行数: 需要统计")
    print(f"过滤后数据条数: {len(filtered_data)}")
    print(f"输出文件: {output_file}")

if __name__ == "__main__":
    filter_and_convert_data()
