import os
from pyspark.sql import SparkSession

def read_local_lance_file(file_path):
    """
    使用 PySpark 和 Lance Spark 连接器在本地读取一个 .lance 文件。
    此版本使用了在 Maven Central 上经过验证的正确坐标。
    """
    print("--- Spark 本地 Lance 文件读取脚本 (最终验证版) ---")

    # --- 1. 定义在 Maven Central 上经过验证的正确坐标 ---
    # Group ID 是 'io.lancedb' 而不是 'com.lancedb'
    lance_spark_package = "io.lancedb:lance-spark-3.4_2.12:0.10.0"

    # --- 2. 检查文件是否存在 ---
    if not os.path.exists(file_path):
        print(f"\n错误: 文件路径不存在: {file_path}")
        return None

    spark = None
    try:
        # --- 3. 创建并配置 SparkSession ---
        print(f"\n正在创建 SparkSession...")
        print(f"将从 Maven Central 下载 Lance 连接器: {lance_spark_package}")
        
        # 这个包就在默认的 Maven Central 仓库里，所以不需要配置额外的仓库
        spark = SparkSession.builder             .appName("LocalLanceReader")             .master("local[*]")             .config("spark.jars.packages", lance_spark_package)             .config("spark.sql.caseSensitive", "true")             .getOrCreate()

        print("SparkSession 创建成功！")

        # --- 4. 读取 Lance 文件 ---
        file_uri = f"file://{os.path.abspath(file_path)}"
        print(f"\n正在从以下 URI 读取 Lance 文件: {file_uri}")
        
        df = spark.read.format("lance").load(file_uri)
        print("文件读取成功！")

        # --- 5. 对数据进行基本分析和展示 ---
        print("\n--- 数据预览与分析 ---")
        print("DataFrame Schema:")
        df.printSchema()

        row_count = df.count()
        print(f"\n文件总行数: {row_count:,}")

        print("\n前5行数据:")
        df.show(5, truncate=False)

        return df

    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")
        print("\n💡 可能的解决方案:")
        print("1. 确保您的服务器可以访问 https://repo1.maven.org (可能需要代理)。")
        print("2. 确认您的 Conda 环境中已正确安装 Java 17。")
        return None
    
    finally:
        # --- 6. 停止 SparkSession，释放资源 ---
        if spark:
            print("\n正在停止 SparkSession...")
            spark.stop()
            print("SparkSession 已停止。")


if __name__ == "__main__":
    # 定义您的本地 Lance 文件路径
    local_lance_path = "/data/code/Oprover/data/github_raw/00cef695-b55c-4c85-bb00-f49683c88582.lance"
    
    result_df = read_local_lance_file(local_lance_path)
    
    if result_df:
        print("\n✅ 脚本成功执行！")
    else:
        print("\n❌ 脚本执行失败。")
