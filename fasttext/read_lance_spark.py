#!/usr/bin/env python3
"""
使用Spark Lance连接器读取Lance文件
基于提供的Spark代码改写的本地版本
"""

import os
import sys
from pathlib import Path

def read_lance_with_spark(lance_file_path):
    """使用Spark Lance连接器读取Lance文件"""
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import lit
        
        print("=== 使用Spark Lance连接器读取 ===")
        
        # 检查文件是否存在
        if not os.path.exists(lance_file_path):
            print(f"错误: 文件不存在 {lance_file_path}")
            return None
        
        # 创建Spark会话（配置JAR路径）
        print("创建Spark会话...")
        jar_path = "/data/code/spark-jars/delta-core_2.12-2.4.0.jar"
        spark = SparkSession.builder \
            .appName('read_lance_local') \
            .config("spark.jars", jar_path) \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.sql.caseSensitive", "true") \
            .getOrCreate()
        
        print("Spark会话创建成功")
        
        # 使用Lance格式读取文件
        print(f"读取Lance文件: {lance_file_path}")
        
        # 将路径转换为file://格式
        file_uri = f"file://{os.path.abspath(lance_file_path)}"
        print(f"文件URI: {file_uri}")
        
        # 由于没有Lance Spark连接器，我们先尝试读取为parquet格式
        # 如果失败，则回退到本地lance库
        try:
            df = spark.read.parquet(file_uri)
        except Exception as e:
            print(f"Spark读取失败: {e}")
            print("回退到本地Lance库...")
            raise Exception("需要使用本地Lance库")
        
        # 添加行ID和其他信息
        df_with_metadata = df.select('*', '_rowid').withColumn('lance_dataset', lit(os.path.basename(lance_file_path)))
        
        print("✓ Lance文件读取成功!")
        
        # 获取基本信息
        print("\n=== 数据集信息 ===")
        row_count = df_with_metadata.count()
        print(f"总行数: {row_count:,}")
        
        # 获取schema
        print(f"\nSchema:")
        df_with_metadata.printSchema()
        
        # 显示前几行数据
        print("\n=== 前5行数据 ===")
        df_with_metadata.show(5, truncate=False)
        
        # 获取列统计信息
        print("\n=== 列统计信息 ===")
        columns = df_with_metadata.columns
        for col in columns[:10]:  # 只显示前10列
            try:
                non_null_count = df_with_metadata.filter(df_with_metadata[col].isNotNull()).count()
                print(f"{col}: 非空值 {non_null_count:,}/{row_count:,}")
            except Exception as e:
                print(f"{col}: 统计失败 - {e}")
        
        # 转换为Pandas DataFrame（小心内存）
        if row_count <= 10000:  # 只有在数据量不大时才转换
            print(f"\n=== 转换为Pandas DataFrame ===")
            try:
                pandas_df = df_with_metadata.limit(1000).toPandas()
                print(f"Pandas DataFrame形状: {pandas_df.shape}")
                
                # 保存样本数据
                import json
                sample_file = "/data/code/Oprover/fasttext/lance_spark_sample.json"
                try:
                    sample_data = pandas_df.head(5).to_dict('records')
                    with open(sample_file, 'w', encoding='utf-8') as f:
                        json.dump(sample_data, f, ensure_ascii=False, indent=2, default=str)
                    print(f"✓ 样本数据已保存到: {sample_file}")
                except Exception as e:
                    print(f"✗ 保存样本数据失败: {e}")
                    
                return pandas_df
                
            except Exception as e:
                print(f"转换为Pandas失败: {e}")
        else:
            print(f"数据量较大({row_count:,}行)，跳过Pandas转换")
        
        # 停止Spark会话
        spark.stop()
        return df_with_metadata
        
    except ImportError as e:
        print(f"✗ PySpark未安装: {e}")
        print("请安装PySpark: pip install pyspark")
        return None
    except Exception as e:
        print(f"✗ Spark读取失败: {e}")
        return None

def read_lance_with_local_tools():
    """使用本地工具尝试读取"""
    print("\n=== 本地工具读取尝试 ===")
    
    lance_file_path = "/data/code/Oprover/data/github_raw/00cef695-b55c-4c85-bb00-f49683c88582.lance"
    
    # 尝试1: 使用lance库
    print("1. 尝试使用lance库...")
    try:
        import lance
        dataset = lance.dataset(lance_file_path)
        print("✓ Lance库读取成功!")
        
        schema = dataset.schema
        print(f"Schema: {schema}")
        
        count = dataset.count_rows()
        print(f"行数: {count:,}")
        
        # 读取数据
        table = dataset.to_table()
        df = table.to_pandas()
        
        print(f"DataFrame形状: {df.shape}")
        print(f"列: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"✗ Lance库失败: {e}")
    
    # 尝试2: 使用lancedb
    print("\n2. 尝试使用lancedb...")
    try:
        import lancedb
        
        parent_dir = os.path.dirname(lance_file_path)
        db = lancedb.connect(parent_dir)
        
        tables = db.table_names()
        print(f"可用表: {tables}")
        
        if tables:
            table_name = tables[0]
            print(f"读取表: {table_name}")
            
            # 尝试不同的访问方式
            try:
                table = db[table_name]
                df = table.limit(1000).to_pandas()
                print(f"✓ LanceDB读取成功! 形状: {df.shape}")
                return df
            except Exception as e:
                print(f"✗ LanceDB表访问失败: {e}")
        
    except Exception as e:
        print(f"✗ LanceDB失败: {e}")
    
    return None

def analyze_file_structure():
    """分析Lance文件结构"""
    lance_file_path = "/data/code/Oprover/data/github_raw/00cef695-b55c-4c85-bb00-f49683c88582.lance"
    
    print("\n=== Lance文件结构分析 ===")
    
    if not os.path.exists(lance_file_path):
        print(f"文件不存在: {lance_file_path}")
        return
    
    file_size = os.path.getsize(lance_file_path)
    print(f"文件大小: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    # 检查是否是目录
    parent_dir = os.path.dirname(lance_file_path)
    print(f"父目录: {parent_dir}")
    
    # 列出目录内容
    try:
        items = os.listdir(parent_dir)
        print(f"目录内容: {items}")
    except Exception as e:
        print(f"无法列出目录内容: {e}")
    
    # 检查文件类型
    with open(lance_file_path, 'rb') as f:
        header = f.read(100)
        f.seek(-100, 2)
        footer = f.read(100)
    
    print(f"文件头部: {header[:50].hex()}")
    print(f"文件尾部: {footer[-50:].hex()}")
    
    # 检查是否包含Lance标识
    if b'LANC' in footer:
        print("✓ 确认Lance格式文件")
    else:
        print("✗ 未发现Lance格式标识")

def main():
    """主函数"""
    lance_file_path = "/data/code/Oprover/data/github_raw/00cef695-b55c-4c85-bb00-f49683c88582.lance"
    
    print("Lance文件读取工具")
    print("="*50)
    
    # 分析文件结构
    analyze_file_structure()
    
    # 方法1: 使用Spark Lance连接器
    result = read_lance_with_spark(lance_file_path)
    
    if result is not None:
        print("\n✅ Spark方法成功!")
        return result
    
    # 方法2: 使用本地工具
    print("\n" + "="*50)
    print("Spark方法失败，尝试本地工具...")
    
    result = read_lance_with_local_tools()
    
    if result is not None:
        print("\n✅ 本地工具成功!")
        return result
    
    print("\n❌ 所有方法都失败了")
    print("\n💡 建议:")
    print("1. 安装PySpark和Lance Spark连接器")
    print("2. 配置正确的Spark环境")
    print("3. 使用原始的Spark集群环境读取")
    
    return None

if __name__ == "__main__":
    result = main()
