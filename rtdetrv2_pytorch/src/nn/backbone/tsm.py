import torch
import torch.nn as nn

class TemporalShift(nn.Module):
    def __init__(self, n_div=8):
        """
        Temporal Shift Module (TSM)

        Args:
            n_div (int): 채널을 몇 그룹으로 나눌지 결정. 
                        (예: 8이면 1/8 채널을 좌/우로 시프트)
        """
        super(TemporalShift, self).__init__()
        self.fold_div = n_div
        self.num_frames = -1  # 비디오 프레임 수 (T), 외부에서 설정됨

    # 학습/추론 시 매번 프레임 수를 설정하기 위한 헬퍼 함수
    def set_num_frames(self, T):
        self.num_frames = T

    def forward(self, x):
        """
        x: (torch.Tensor) 입력 피처맵.
            Shape:
            여기서 B = N * T (배치 크기 * 프레임 수)
        """
        # num_frames가 설정되지 않았으면 (예: 단일 이미지 추론) 그냥 통과
        if self.num_frames == -1:
            return x

        B, C, H, W = x.shape
        T = self.num_frames

        # B가 T로 나누어 떨어지는지 확인
        if B % T!= 0:
            raise ValueError(f"Batch size ({B}) is not divisible by num_frames ({T})")

        N = B // T  # 실제 배치 크기 (N)

        # 1. -> (5D 텐서로 변환)
        x = x.view(N, T, C, H, W)

        fold = C // self.fold_div
        out = torch.zeros_like(x)

        # 2. 채널 시프트 수행 [1]
        # (t+1) 프레임의 1/8 채널을 (t) 프레임으로 이동 (좌측 시프트)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        # (t-1) 프레임의 1/8 채널을 (t) 프레임으로 이동 (우측 시프트)
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        # 나머지 채널(6/8)은 그대로 유지
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        # 3. -> (다시 4D 텐서로 변환)
        return out.view(B, C, H, W)