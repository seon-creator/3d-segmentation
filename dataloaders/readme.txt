# dataloader
build_dataset.py
- 데이터셋 타입을 설정하면 그에 맞는 dataloader를 불러오도록 구현
- 사용 예시
config.yaml 코드 중 데이터로터 부분:
# dataset args
dataset_parameters:
  dataset_type: "brats_seg" <-- 이 부분에 해당

brats.py
- BraTS 데이터셋을 불러오는 코드

