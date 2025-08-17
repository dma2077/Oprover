import os
import json
import time
import argparse
import pandas as pd
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

from model import APIModel, default_model
from dataset import DifficultyDataset


class DifficultyInference:
    """难度评估推理引擎，负责读取数据、调用模型、保存结果"""
    
    def __init__(self, model: Optional[APIModel] = None, dataset: Optional[DifficultyDataset] = None):
        """
        初始化推理引擎
        
        Args:
            model: API模型
            dataset: 数据集处理器
        """
        self.model = model or default_model
        self.dataset = dataset or DifficultyDataset()
    
    def process_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个数据项
        
        Args:
            item: 数据项，包含uuid和formal_statement
            
        Returns:
            处理后的数据项
        """
        uuid = item.get("uuid")
        formal_statement = item.get("formal_statement", "").strip()
        
        if not formal_statement:
            return {
                "uuid": uuid,
                "formal_statement": formal_statement,
                "model_output": None,
                "difficulty_score": None,
                "success": False,
                "error": "Empty formal statement"
            }
        
        try:
            # 构建提示
            prompt = self.dataset.build_prompt(formal_statement)
            
            # 调用模型预测
            prediction = self.model.predict(
                prompt=prompt,
                expected_keys=self.dataset.get_expected_keys(),
                validate_fn=self.dataset.get_validation_function()
            )
            
            if prediction.get("status") == "success" and prediction.get("data"):
                data = prediction["data"]
                difficulty_score = data.get("Difficulty")
                
                return {
                    "uuid": uuid,
                    "formal_statement": formal_statement,
                    "model_output": prediction.get("response"),
                    "difficulty_score": difficulty_score,
                    "success": True,
                    "error": None
                }
            else:
                return {
                    "uuid": uuid,
                    "formal_statement": formal_statement,
                    "model_output": prediction.get("response"),
                    "difficulty_score": None,
                    "success": False,
                    "error": prediction.get("error", "Prediction failed")
                }
                
        except Exception as e:
            return {
                "uuid": uuid,
                "formal_statement": formal_statement,
                "model_output": None,
                "difficulty_score": None,
                "success": False,
                "error": str(e)
            }
    
    def process_batch(self, batch: List[Dict[str, Any]], max_workers: int = 8) -> List[Dict[str, Any]]:
        """
        批量处理数据
        
        Args:
            batch: 数据批次
            max_workers: 最大工作线程数
            
        Returns:
            处理后的数据批次
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_single_item, item): item for item in batch}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batch"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing item: {e}")
                    # 保留原始数据项
                    original_item = futures[future]
                    results.append({
                        "uuid": original_item.get("uuid"),
                        "formal_statement": original_item.get("formal_statement"),
                        "model_output": None,
                        "difficulty_score": None,
                        "success": False,
                        "error": str(e)
                    })
        
        return results
    
    def create_batch(self, data: List[Dict[str, Any]], batch_size: int = 1) -> List[List[Dict[str, Any]]]:
        """
        创建批次数据
        
        Args:
            data: 原始数据
            batch_size: 批次大小
            
        Returns:
            批次数据列表
        """
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        return batches


def main():
    """主函数，处理命令行参数并执行推理任务"""
    parser = argparse.ArgumentParser(description="难度评估推理引擎")
    parser.add_argument("--input", type=str, required=True, 
                       help="输入parquet文件路径")
    parser.add_argument("--output", type=str, default="results.jsonl", 
                       help="输出JSONL文件路径")
    parser.add_argument("--workers", type=int, default=8, 
                       help="最大工作线程数")
    parser.add_argument("--batch-size", type=int, default=100, 
                       help="批次大小")
    parser.add_argument("--max-items", type=int, default=None, 
                       help="最大处理项目数（用于测试）")
    
    args = parser.parse_args()
    
    # 创建推理引擎
    dataset = DifficultyDataset()
    inference = DifficultyInference(dataset=dataset)
    output_path = Path(args.output)
    
    print(f"🚀 启动难度评估推理引擎")
    print(f"📁 输入文件: {args.input}")
    print(f"📁 输出文件: {args.output}")
    print(f"🔧 工作线程: {args.workers}")
    print(f"📦 批次大小: {args.batch_size}")
    
    try:
        # 读取parquet文件
        print(f"\n📊 读取数据文件...")
        df = pd.read_parquet(args.input)
        print(f"✅ 成功读取 {len(df)} 条数据")
        
        # 检查必要的列
        required_columns = ["uuid", "formal_statement"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ 缺少必要的列: {missing_columns}")
            print(f"📋 可用列: {list(df.columns)}")
            return 1
        
        # 转换为字典列表
        data = []
        for _, row in df.iterrows():
            data.append({
                "uuid": row["uuid"],
                "formal_statement": row["formal_statement"]
            })
        
        # 如果输出文件存在，则读取其中已成功处理过的 uuid，并在本次推理中跳过
        processed_success_uuid_set = set()
        if output_path.exists():
            print(f"🧮 检查已存在的输出文件，载入已完成的UUID以跳过...")
            loaded_lines = 0
            loaded_success = 0
            try:
                with output_path.open('r', encoding='utf-8') as fout:
                    for line in fout:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        loaded_lines += 1
                        if obj.get('success') is True:
                            uuid_val = obj.get('uuid')
                            if isinstance(uuid_val, str):
                                processed_success_uuid_set.add(uuid_val)
                                loaded_success += 1
            except Exception as e:
                print(f"⚠️ 读取输出文件失败，忽略跳过逻辑: {e}")
            print(f"✅ 已读取历史 {loaded_lines} 行，其中成功 {loaded_success} 条，将跳过 {len(processed_success_uuid_set)} 个UUID")
        
        if processed_success_uuid_set:
            before = len(data)
            data = [item for item in data if item.get('uuid') not in processed_success_uuid_set]
            after = len(data)
            print(f"⏭️ 跳过已完成的 {before - after} 条，剩余待处理 {after} 条")
        
        # 限制处理数量（用于测试）
        if args.max_items:
            data = data[:args.max_items]
            print(f"🔧 限制处理数量为: {len(data)} 条")
        
        print(f"📊 准备处理 {len(data)} 条数据")
        
        # 创建批次
        batches = inference.create_batch(data, args.batch_size)
        print(f"📦 分为 {len(batches)} 个批次处理")
        
        # 处理所有批次
        processed_count = 0
        success_count = 0
        
        # 确保输出文件存在（不清空，采用追加模式继续写入）
        if not output_path.exists():
            print(f"\n💾 初始化输出文件: {args.output}")
            with output_path.open('w', encoding='utf-8') as f:
                pass
        else:
            print(f"\n💾 继续写入已有的输出文件（追加模式）: {args.output}")
        
        for i, batch in enumerate(batches):
            print(f"\n🔁 处理批次 {i+1}/{len(batches)} ({len(batch)} 条数据)")
            
            # 处理批次
            results = inference.process_batch(batch, args.workers)
            
            # 立即保存批次结果
            print(f"💾 保存批次 {i+1} 结果...")
            with output_path.open('a', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            # 统计
            processed_count += len(results)
            batch_success = len([r for r in results if r["success"]])
            success_count += batch_success
            
            print(f"✅ 批次 {i+1} 完成: 处理 {len(results)} 条，成功 {batch_success} 条，已保存")
            
            # 添加延迟避免API限制
            time.sleep(1)
        
        # 最终统计
        stats = {
            "status": "completed",
            "total": len(data),
            "processed": processed_count,
            "success": success_count,
            "success_rate": success_count / processed_count if processed_count > 0 else 0
        }
        
        print(f"\n🎉 任务完成!")
        print(f"📈 统计信息: {stats}")
        print(f"📁 所有结果已保存到: {args.output}")
        
    except Exception as e:
        print(f"\n❌ 任务失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
