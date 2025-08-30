#!/usr/bin/env python3
"""
Multi-Round Feedback Inference Script (Simplified)
简化的多轮反馈推理脚本 - 只负责验证和推理逻辑

This script handles inference and validation logic only.
Server deployment is handled by the bash script.
"""

import os
import sys
import subprocess
import time
import json
import requests
import argparse
from pathlib import Path

class MultiRoundInference:
    def __init__(self, model_name, dataset_name, split_num, prompt_config, max_rounds):
        # 基本参数
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.split_num = f"{int(split_num):02d}"
        self.prompt_config = prompt_config
        self.max_rounds = int(max_rounds)
        
        # 路径配置
        self.oprover_dir = Path("/data/code/Oprover")
        self.data_root_dir = Path("/madehua/data/oprover/generated_data")
        self.validation_service_dir = Path("/data/code/kimina-lean-server/server/proof")
        
        # 创建必要目录
        self.output_dir = self.data_root_dir / self.dataset_name / f"{self.model_name}_results"
        self.validation_dir = self.output_dir  # 验证结果也放在模型特定目录下
        self.log_dir = self.output_dir / "logs"  # 日志放在模型特定目录的logs子目录下
        
        for dir_path in [self.output_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 验证服务配置
        self.validation_server_port = 8002
        self.validation_script = self.validation_service_dir / "lean_proof_single.py"
        
        # 推理配置
        self.config_file = self.get_model_config()
        self.log_prefix = self.get_log_prefix()
        self.batch_size = "3000"
        
        # 初始数据集路径
        self.initial_split = f"{self.dataset_name}/lean_statement_part_{self.split_num}"
    
    def get_model_config(self):
        """根据模型名称返回配置文件路径"""
        config_map = {
            "DeepSeek-Prover-V2-7B": "config/config_dpsk.yaml",
            "Goedel-Prover-V2-8B": "config/config_goedel.yaml", 
            "Goedel-Prover-V2-32B": "config/config_goedel.yaml",
            "Kimina-Prover-72B": "config/config_kimina.yaml"
        }
        return config_map.get(self.model_name, "config/config_dpsk.yaml")
    
    def get_log_prefix(self):
        """根据模型名称返回日志前缀"""
        prefix_map = {
            "DeepSeek-Prover-V2-7B": "dpsk",
            "Goedel-Prover-V2-8B": "g8",
            "Goedel-Prover-V2-32B": "g32", 
            "Kimina-Prover-72B": "k72"
        }
        if self.model_name in prefix_map:
            return prefix_map[self.model_name]
        else:
            import re
            simplified = re.sub(r'[^a-z0-9]', '', self.model_name.lower())
            return simplified[:10]
    
    def check_service_health(self):
        """检查验证服务是否运行"""
        try:
            response = requests.get(f"http://localhost:{self.validation_server_port}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def run_inference_round(self, round_num, input_split, round_output_dir):
        """运行单轮推理"""
        print(f"=== Round {round_num}: Starting inference ===")
        print(f"Input split: {input_split}")
        print(f"Output dir: {round_output_dir}")
        
        # 创建输出目录
        Path(round_output_dir).mkdir(parents=True, exist_ok=True)
        
        # 构建推理命令
        log_file = self.log_dir / f"{self.log_prefix}_{self.prompt_config}_round{round_num}_worker_{self.split_num}.log"
        
        cmd = [
            "python", "infer/infer.py",
            "--config", self.config_file,
            "--split", input_split,
            "--mode", f"{self.prompt_config}-bon",
            "--model_name", self.model_name,
            "--output_dir", round_output_dir,
            "--batch_size", self.batch_size,
            "--use_accel",
            "--index", "0", 
            "--world_size", "1"
        ]
        
        try:
            print(f"Running command: {' '.join(cmd)}")
            
            # 设置环境变量
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.oprover_dir)
            
            # 为data_loader.py设置输入文件路径
            if round_num == 1:
                # 第1轮使用原始数据
                env['INPUT_FILE'] = str(self.initial_split)
            else:
                # 后续轮次使用上一轮的验证结果
                prev_validation_file = self.validation_dir / f"part_{self.split_num}_round{round_num-1}_proof.jsonl"
                env['INPUT_FILE'] = str(prev_validation_file)
            
            # 设置输出文件路径，让infer.py直接输出到期望的位置
            inference_output_file = self.output_dir / f"part_{self.split_num}_round{round_num}_inference.jsonl"
            env['OUTPUT_FILE'] = str(inference_output_file)
            
            # 运行推理
            with open(log_file, 'w') as f:
                process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=self.oprover_dir)
            
            # 不依赖返回码，而是检查是否生成了目标文件
            # 现在infer.py会直接输出到期望的路径
            expected_output_file = self.output_dir / f"part_{self.split_num}_round{round_num}_inference.jsonl"
            
            if expected_output_file.exists():
                print(f"✅ Round {round_num} inference completed successfully (output file found)")
                return True
            else:
                print(f"❌ Round {round_num} inference failed (no output file generated)")
                print(f"Expected output file: {expected_output_file}")
                return False
            
        except Exception as e:
            print(f"❌ Error in round {round_num} inference: {e}")
            return False
    
    def run_validation(self, round_num, inference_output_file, validation_output_file):
        """运行验证"""
        print(f"=== Round {round_num}: Starting validation ===")
        print(f"Inference output: {inference_output_file}")
        print(f"Target validation output: {validation_output_file}")
        
        # 检查验证服务是否运行
        if not self.check_service_health():
            print("❌ Validation service is not running")
            return False
        
        try:
            # 设置环境变量
            env = os.environ.copy()
            env['INPUT_FILE'] = str(inference_output_file)
            
            # 运行验证脚本，传递完整的输出文件路径
            log_file = self.log_dir / f"validation_round{round_num}_{self.split_num}.log"
            
            cmd = [
                "python", str(self.validation_script),
                "--output_file", str(validation_output_file)
            ]
            
            print(f"Running validation command: {' '.join(cmd)}")
            
            with open(log_file, 'w') as f:
                process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=self.validation_service_dir)
            
            if process.returncode == 0:
                # 验证脚本应该直接输出到指定文件路径
                if Path(validation_output_file).exists():
                    print(f"✅ Round {round_num} validation completed successfully")
                    print(f"Validation results saved to: {validation_output_file}")
                else:
                    print(f"❌ Validation output file not found at: {validation_output_file}")
                    return False
            else:
                print(f"❌ Round {round_num} validation failed with return code: {process.returncode}")
            
            return process.returncode == 0
            
        except Exception as e:
            print(f"❌ Error in round {round_num} validation: {e}")
            return False
    
    def check_validation_results(self, validation_file, round_num):
        """检查验证结果，返回失败案例数"""
        print(f"=== Round {round_num} Validation Results ===")
        
        try:
            failed_count = 0
            success_count = 0
            total_count = 0
            
            with open(validation_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        total_count += 1
                        if data.get('success', False):
                            success_count += 1
                        else:
                            failed_count += 1
                    except json.JSONDecodeError:
                        continue
            
            # 计算正确率
            if total_count > 0:
                accuracy = (success_count / total_count) * 100
                print(f"📊 Round {round_num} Statistics:")
                print(f"   ✅ Successful: {success_count}")
                print(f"   ❌ Failed: {failed_count}")
                print(f"   📈 Total: {total_count}")
                print(f"   🎯 Accuracy: {accuracy:.2f}%")
            else:
                print(f"⚠️  No validation results found in round {round_num}")
            
            print("=" * 40)
            return failed_count
            
        except Exception as e:
            print(f"❌ Error checking validation results: {e}")
            return 0
    
    def run(self):
        """运行多轮推理主循环"""
        print("Starting multi-round inference process...")
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Split: {self.split_num}")
        print(f"Prompt Config: {self.prompt_config}")
        print(f"Max Rounds: {self.max_rounds}")
        print("=" * 50)
        
        current_split = self.initial_split
        
        for round_num in range(1, self.max_rounds + 1):
            print()
            print("/" * 50)
            print(f"/// ROUND {round_num} / {self.max_rounds}")
            print("/" * 50)
            
            # 设置当前轮次的输出文件路径
            # 现在infer.py会直接输出到期望的路径
            inference_output_file = self.output_dir / f"part_{self.split_num}_round{round_num}_inference.jsonl"
            # 简化的验证文件名（包含轮次）
            validation_output_file = self.validation_dir / f"part_{self.split_num}_round{round_num}_proof.jsonl"
            
            # 第1步：运行推理
            print(f"Starting round {round_num} inference...")
            
            try:
                self.run_inference_round(round_num, current_split, str(self.output_dir))
            except Exception as e:
                print(f"⚠️  Round {round_num} inference encountered an error: {e}")
                print(f"⏭️  Continuing to check if output file was generated...")
            
            # 检查推理脚本生成的输出文件
            if not inference_output_file.exists():
                print(f"❌ Round {round_num} failed: No inference output file generated")
                print(f"Expected file: {inference_output_file}")
                print(f"⏭️  Skipping round {round_num} and continuing...")
                return False
            
            print(f"✅ Round {round_num} inference result saved: {inference_output_file.name}")
            
            # 第2步：运行验证
            if not self.run_validation(round_num, str(inference_output_file), str(validation_output_file)):
                print(f"❌ Round {round_num} validation failed. Stopping.")
                return False
            
            # 检查验证输出文件是否存在
            if not validation_output_file.exists():
                print(f"❌ Validation output file not found: {validation_output_file}")
                return False
            
            # 第3步：检查验证结果
            failed_count = self.check_validation_results(str(validation_output_file), round_num)
            
            # 如果不是最后一轮，检查是否需要继续
            if round_num < self.max_rounds:
                if failed_count == 0:
                    print("🎉 All problems solved successfully! No need for further rounds.")
                    break
                
                # 设置下一轮的输入（使用类似 FineLeanCorpus/lean_statement_part_00 的split格式）
                # data_loader.py会自动过滤掉成功的案例
                current_split = f"{self.dataset_name}/lean_statement_part_{self.split_num}"
                print(f"➡️  Next round will use: split={current_split}")
                print(f"   (Input file: {validation_output_file})")
        
        print()
        print("/" * 50)
        print("/// MULTI-ROUND PROCESS COMPLETED")
        print("/" * 50)
        
        # 最终汇总统计
        print("📊 FINAL RESULTS SUMMARY")
        print("=" * 50)
        
        all_rounds_stats = []
        for round_num in range(1, self.max_rounds + 1):
            validation_file = self.validation_dir / f"part_{self.split_num}_round{round_num}_proof.jsonl"
            if validation_file.exists():
                try:
                    success_count = 0
                    failed_count = 0
                    total_count = 0
                    with open(validation_file, 'r') as f:
                        for line in f:
                            try:
                                data = json.loads(line.strip())
                                total_count += 1
                                if data.get('success', False):
                                    success_count += 1
                                else:
                                    failed_count += 1
                            except json.JSONDecodeError:
                                continue
                    
                    if total_count > 0:
                        accuracy = (success_count / total_count) * 100
                        all_rounds_stats.append((round_num, success_count, failed_count, total_count, accuracy))
                        print(f"Round {round_num}: {success_count}/{total_count} successful ({accuracy:.1f}%)")
                    else:
                        print(f"Round {round_num}: No results")
                except Exception as e:
                    print(f"Round {round_num}: Error reading results - {e}")
        
        print("=" * 50)
        
        print()
        print(f"All output files (inference + validation) saved in: {self.output_dir}")
        print(f"All logs saved in: {self.log_dir}")
        print("✅ Multi-round feedback process completed successfully!")
        
        return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Multi-Round Feedback Inference Script')
    parser.add_argument('--model_name', type=str, default='DeepSeek-Prover-V2-7B', help='Model name')
    parser.add_argument('--dataset_name', type=str, default='FineLeanCorpus', help='Dataset name')
    parser.add_argument('--split_num', type=str, default='00', help='Split number')
    parser.add_argument('--prompt_config', type=str, default='proof_cot_feedback', help='Prompt configuration')
    parser.add_argument('--max_rounds', type=str, default='3', help='Maximum number of rounds')
    
    args = parser.parse_args()
    
    try:
        runner = MultiRoundInference(
            args.model_name, 
            args.dataset_name, 
            args.split_num, 
            args.prompt_config, 
            args.max_rounds
        )
        success = runner.run()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())