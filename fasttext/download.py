import os
import boto3
from tqdm import tqdm
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor

class SimpleDirectoryDownloader:
    def __init__(self, client, bucket, max_workers=4):
        self.client = client
        self.bucket = bucket
        self.max_workers = max_workers

    def download_file_with_progress(self, s3_key, local_file):
        """下载单个文件并显示进度"""
        try:
            # 确保目标目录存在
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            
            # 获取文件大小
            response = self.client.head_object(Bucket=self.bucket, Key=s3_key)
            file_size = response['ContentLength']
            
            with tqdm(total=file_size, unit='B', unit_scale=True,
                     desc=f"Downloading {os.path.basename(local_file)}") as pbar:
                
                def callback(bytes_amount):
                    pbar.update(bytes_amount)

                self.client.download_file(
                    self.bucket,
                    s3_key,
                    local_file,
                    Callback=callback
                )
            return True
        except Exception as e:
            print(f"\n下载 {s3_key} 失败: {e}")
            return False

    def download_directory(self, s3_prefix, local_dir):
        """下载目录"""
        # 扫描S3中的文件
        files_to_download = []
        total_size = 0
        print("扫描S3目录中...")

        paginator = self.client.get_paginator('list_objects_v2')
        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix):
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    # 确保文件在指定前缀下
                    if not s3_key.startswith(s3_prefix):
                        continue
                        
                    # 计算本地路径
                    relative_path = s3_key[len(s3_prefix):].lstrip('/')
                    local_path = os.path.join(local_dir, relative_path)
                    
                    # 检查本地文件
                    if os.path.exists(local_path):
                        local_size = os.path.getsize(local_path)
                        if local_size == obj['Size']:
                            print(f"跳过 {s3_key} (大小相同)")
                            continue
                            
                    files_to_download.append((s3_key, local_path))
                    total_size += obj['Size']
                    
        except ClientError as e:
            print(f"扫描S3出错: {e}")
            return

        if not files_to_download:
            print("没有文件需要下载！")
            return

        print(f"\n需要下载 {len(files_to_download)} 个文件，总大小: {total_size / (1024*1024):.2f} MB")

        # 使用线程池下载文件
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for s3_key, local_path in files_to_download:
                future = executor.submit(
                    self.download_file_with_progress,
                    s3_key,
                    local_path
                )
                futures.append((future, s3_key))

            # 等待所有下载完成
            for future, s3_key in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"\n处理 {s3_key} 时发生错误: {e}")

        print("\n下载完成！")

# 使用示例
if __name__ == "__main__":
    # 配置客户端
    client = boto3.client(
        's3',
        endpoint_url='http://relay.fortidyndns.com:10086',
        aws_access_key_id='MAKI3JWDDQV7YP4QUAH6',
        aws_secret_access_key='RPDbNSUWAS6ClY7luCalUAhiVKqtRCgwkpb6DKG6',
        region_name='RegionOne',
        use_ssl=False
    )
    # 创建下载器实例
    downloader = SimpleDirectoryDownloader(
        client=client,
        bucket='2077ai',
        max_workers=16  # 同时下载的文件数
    )
    try:
        # 下载目录1
        downloader.download_directory(
            s3_prefix='M-A-P/shuyue/Prover/fasttext/code_lean/',  # S3中的路径
            local_dir='/madehua/model/fasttext'  # 本地保存路径
        )
    except KeyboardInterrupt:
        print("\n下载已暂停，下次运行时将从断点继续")