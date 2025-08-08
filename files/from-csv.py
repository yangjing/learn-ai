import argparse
import os
import csv
from urllib.parse import urlparse
from subprocess import run, CalledProcessError
from datetime import datetime
import obs
import sys

DOWNLOADS_DIR = "files"


# 不再需要此函数，因为我们直接使用 OBS SDK
# 保留此函数以保持向后兼容
def to_obs_url(url):
  # 解析 URL
  parsed_url = urlparse(url)
  # 提取 bucket 名称
  bucket_name = parsed_url.netloc.split(".")[0]
  # 提取路径部分
  path = parsed_url.path
  # 构造新的 URL 格式
  new_url = f"obs://{bucket_name}{path}"
  return new_url


def _load_progress_last(progress_file: str) -> str | None:
  try:
    with open(progress_file, "r", newline="") as f:
      reader = csv.DictReader(f)
      last = None
      for row in reader:
        last = row
      return last["id"] if last else None
  except FileNotFoundError:
    return None


def _save_progress(progress_file, status, file_id, name, file_url, create_time):
  with open(progress_file, "a", newline="") as f:
    fieldnames = ["status", "id", "name", "file_url", "create_time"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if f.tell() == 0:  # 如果文件为空，写入表头
      writer.writeheader()
    writer.writerow(
      {
        "status": status,
        "id": file_id,
        "name": name,
        "file_url": file_url,
        "create_time": create_time,
      }
    )


class FileDownloader:
  def __init__(self, csv_file):
    # 动态生成 progress_file 名称
    progress_file = os.path.splitext(csv_file)[0] + ".log"
    self.csv_file = csv_file
    self.progress_file = progress_file
    self.downloaded = _load_progress_last(progress_file)
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)

  def _create_local_path(self, hostname, url_path):
    path_parts = url_path.split("/")[:-1]  # 排除文件名
    local_dir = os.path.join(DOWNLOADS_DIR, hostname, *path_parts)
    os.makedirs(local_dir, exist_ok=True)
    return os.path.join(local_dir, os.path.basename(url_path))

  def _download_file(self, file_data):
    file_id, file_url, name = (
      file_data["id"],
      file_data["file_url"],
      file_data["name"],
    )
    if not file_url:
      return

    hostname, url_path = self._parse_url(file_url)
    local_path = self._create_local_path(hostname, url_path)
    create_time = datetime.now().isoformat()

    status = "F"

    try:
      if "obs.cn-southwest-2.myhuaweicloud.com" in file_url:
        # 使用华为云 OBS SDK 下载（需提前配置好ak/sk）
        # 从环境变量或配置文件获取 AK/SK
        access_key_id = os.environ.get("OBS_ACCESS_KEY_ID")
        secret_access_key = os.environ.get("OBS_SECRET_ACCESS_KEY")
        server = urlparse(file_url).netloc

        if not access_key_id or not secret_access_key:
          print("警告: 未设置 OBS_ACCESS_KEY_ID 或 OBS_SECRET_ACCESS_KEY 环境变量")
          print("请设置环境变量或在代码中直接配置 AK/SK")
          # 如果环境变量未设置，可以在这里直接设置
          # access_key_id = '你的AK'
          # secret_access_key = '你的SK'

        # 创建 OBS 客户端
        obs_client = obs.ObsClient(access_key_id=access_key_id, secret_access_key=secret_access_key, server=server)

        # 解析 URL 获取桶名和对象键
        bucket_name = urlparse(file_url).netloc.split(".")[0]
        object_key = urlparse(file_url).path.lstrip("/")
        
        # 检查本地文件是否存在
        need_download = True
        if os.path.exists(local_path):
          # 获取本地文件大小
          local_file_size = os.path.getsize(local_path)
          
          # 获取 OBS 对象元数据
          resp_meta = obs_client.getObjectMetadata(bucketName=bucket_name, objectKey=object_key)
          
          if resp_meta.status < 300 and 'content-length' in resp_meta.header:
            # 获取 OBS 对象大小
            obs_file_size = int(resp_meta.header['content-length'])
            
            # 比较文件大小
            if local_file_size == obs_file_size:
              need_download = False
              status = "T"
              print(f"跳过下载: {file_id} -> {local_path} (文件已存在且大小一致)")
        
        # 只有需要下载时才执行下载
        if need_download:
          # 下载对象
          resp = obs_client.getObject(bucketName=bucket_name, objectKey=object_key, downloadPath=local_path)

          if resp.status < 300:
            status = "T"
            print(f"下载成功: {file_id} -> {local_path}")
          else:
            print(f"下载失败: {file_id} -> {local_path}, 状态码: {resp.status}, 错误: {resp.errorCode}")
      else:
        # 通过 wget 下载
        result = run(["wget", "--no-check-certificate", "-O", local_path, file_url], check=True)
        if result.returncode == 0:
          status = "T"
          print(f"下载成功: {file_id} -> {local_path}")
        else:
          print(f"下载失败：[{result.returncode}] {file_id} -> {local_path}")
    except Exception as e:
      print(f"下载失败: {file_id} -> {local_path} {e}")

    _save_progress(self.progress_file, status, file_id, name, file_url, create_time)
    return status

  def _parse_url(self, file_url):
    """
    解析文件 URL，提取主机名和路径。
    :param file_url: 文件的完整 URL
    :return: (hostname, url_path)
    """
    parsed_url = urlparse(file_url)
    hostname = parsed_url.netloc
    url_path = parsed_url.path
    return hostname, url_path

  def process_files(self):
    # 从 CSV 文件中读取数据
    with open(self.csv_file, "r", newline="") as f:
      reader = csv.DictReader(f)
      for row in reader:
        file_id = row["id"]

        # 跳过已下载的文件
        if self.downloaded and file_id <= self.downloaded:
          continue

        # 直接调用下载方法
        self._download_file(row)


if __name__ == "__main__":
  # 添加命令行参数解析
  parser = argparse.ArgumentParser(description="下载 CSV 文件中列出的文件。")
  parser.add_argument("csv_file", type=str, help="包含文件下载信息的 CSV 文件路径")
  args = parser.parse_args()

  downloader = FileDownloader(args.csv_file)
  downloader.process_files()
