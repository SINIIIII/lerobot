import os
import json
from safetensors.torch import load_file, save_file

# 에러 로그에 나온 모델 캐시 폴더의 정확한 경로
cache_dir = "/home/semoon/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e"
index_path = os.path.join(cache_dir, "model.safetensors.index.json")

print("1. 인덱스 파일 읽는 중...")
with open(index_path, "r") as f:
    index = json.load(f)

weight_map = index["weight_map"]
files_to_load = set(weight_map.values())

full_state_dict = {}
for file in files_to_load:
    part_path = os.path.join(cache_dir, file)
    print(f"2. 조각 파일 불러오는 중: {file} ...")
    part_state = load_file(part_path)
    full_state_dict.update(part_state)

save_path = os.path.join(cache_dir, "model.safetensors")
print(f"3. 단일 파일로 병합 저장 중 (잠시만 기다려주세요): {save_path} ...")
save_file(full_state_dict, save_path)
print("✅ 병합 완료! 이제 모델이 인식할 수 있습니다.")