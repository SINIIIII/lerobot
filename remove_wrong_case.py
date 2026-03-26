import os
import json
import glob
import shutil
from huggingface_hub import snapshot_download, HfApi
from datasets import load_dataset

# 설정
repo_id = "sini-21/SO101-record-grip-cylinder"
episodes_to_remove = {2, 5, 6, 9, 22, 23, 27, 35, 53, 54}

print("1. 데이터셋 다운로드 중 (무거운 영상 파일은 제외)...")
download_dir = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=["data/**/*.parquet", "meta/**/*.parquet", "meta/*.json"],
    local_dir="./dataset_download"
)

upload_dir = "./dataset_upload"
os.makedirs(os.path.join(upload_dir, "meta/episodes/chunk-000"), exist_ok=True)
os.makedirs(os.path.join(upload_dir, "data/chunk-000"), exist_ok=True)

print("2. 에피소드 데이터(meta/episodes) 필터링 및 재정렬 중...")
ep_files = glob.glob(os.path.join(download_dir, "meta/episodes/**/*.parquet"), recursive=True)
ep_dataset = load_dataset("parquet", data_files={"train": ep_files}, split="train")

# 삭제할 에피소드 걸러내기
ep_filtered = ep_dataset.filter(lambda x: x["episode_index"] not in episodes_to_remove, load_from_cache_file=False)

# 새로운 인덱스 매핑 생성
old_to_new_ep = {}
for new_idx, old_idx in enumerate(ep_filtered["episode_index"]):
    old_to_new_ep[old_idx] = new_idx

def update_ep(batch):
    batch["episode_index"] = [old_to_new_ep[ep] for ep in batch["episode_index"]]
    return batch

ep_updated = ep_filtered.map(update_ep, batched=True, load_from_cache_file=False)
ep_updated.to_parquet(os.path.join(upload_dir, "meta/episodes/chunk-000/file-000.parquet"))

print("3. 프레임 데이터(data/) 필터링 및 인덱스 재정렬 중...")
data_files = glob.glob(os.path.join(download_dir, "data/**/*.parquet"), recursive=True)
data_dataset = load_dataset("parquet", data_files={"train": data_files}, split="train")

data_dataset = data_dataset.sort("index")
data_filtered = data_dataset.filter(lambda x: x["episode_index"] not in episodes_to_remove, load_from_cache_file=False)

def update_data(batch, indices):
    batch["episode_index"] = [old_to_new_ep[ep] for ep in batch["episode_index"]]
    batch["index"] = indices
    return batch

data_updated = data_filtered.map(update_data, with_indices=True, batched=True, load_from_cache_file=False)
data_updated.to_parquet(os.path.join(upload_dir, "data/chunk-000/file-000.parquet"))

print("4. info.json 및 기타 메타데이터 업데이트 중...")
with open(os.path.join(download_dir, "meta/info.json"), "r") as f:
    info = json.load(f)

info["total_episodes"] = len(ep_updated)
info["total_frames"] = len(data_updated)

# 문자열(String) 타입과 딕셔너리(Dict) 타입을 모두 지원하도록 수정
if "splits" in info and "train" in info["splits"]:
    if isinstance(info["splits"]["train"], str):
        # LeRobot v2의 문자열 포맷 ("0:60" -> "0:50")
        info["splits"]["train"] = f"0:{len(ep_updated)}"
    elif isinstance(info["splits"]["train"], dict):
        # 일반 포맷
        info["splits"]["train"]["num_episodes"] = len(ep_updated)
        info["splits"]["train"]["num_frames"] = len(data_updated)

with open(os.path.join(upload_dir, "meta/info.json"), "w") as f:
    json.dump(info, f, indent=4)

shutil.copy(os.path.join(download_dir, "meta/stats.json"), os.path.join(upload_dir, "meta/stats.json"))
if os.path.exists(os.path.join(download_dir, "meta/tasks.parquet")):
    shutil.copy(os.path.join(download_dir, "meta/tasks.parquet"), os.path.join(upload_dir, "meta/tasks.parquet"))

print("5. 허깅페이스 원본 레포지토리에 덮어쓰기 중...")
api = HfApi()

try:
    api.delete_folder(repo_id=repo_id, repo_type="dataset", path_in_repo="data")
    api.delete_folder(repo_id=repo_id, repo_type="dataset", path_in_repo="meta/episodes")
except Exception:
    pass 

# 👉 이 부분이 api.upload 로 되어있을 텐데, 아래처럼 api.upload_folder 로 변경해 주세요!
api.upload_folder(
    folder_path=upload_dir,
    repo_id=repo_id,
    repo_type="dataset"
)

print("✨ 드디어 모든 작업이 완벽하게 끝났습니다!")