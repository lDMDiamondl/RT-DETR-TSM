# RT-DETR + TSM

이 저장소는 [RT-DETR](https://github.com/lyuwenyu/RT-DETR)을 기반으로, 백본에 [Temporal Shift Module (TSM)](https://arxiv.org/abs/1811.08383)을 추가한 코드입니다.  
본 변경은 영상에서 실시간 객체 탐지 성능 향상을 위한 시간적 정보 활용 연구를 목적으로 합니다.

## 특징

- 영상 프레임에서 실시간 객체 탐지
- TSM을 통한 시간적 특징 모델링
- 실험용으로 유연하게 사용 가능

## TODO

- **백본 수정:** `src/nn/backbone/presnet.py`  
- **모델 수정:** `src/zoo/rtdetr/rtdetr.py`  
- **데이터셋 추가:** `src/data/dataset/new_video_dataset.py`