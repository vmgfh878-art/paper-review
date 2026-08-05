"""
CNN-LSTM 정답본 (1주차)

원본: lens/ai/models/cnn_lstm.py + lens/ai/models/blocks.py
논문 핵심 골격만 남기고 축약. (출력 헤드/fp32/티커임베딩 제거)

    python 01-cnn-lstm/code/run_shapes.py --mode solution

shape 흐름:
    [B, 120, 36] --permute(0,2,1)--> [B, 36, 120] --conv x4--> [B, 64, 120]
    --permute(0,2,1)--> [B, 120, 64] --LSTM--> [B, 120, 128]
    --attn pool--> [B, 128] --head--> [B, 5]

축 번호 치트시트:
    x.shape = [ 4 ,  120 ,  36 ]
                ↑     ↑      ↑
              dim=0  dim=1  dim=2
              배치    시간    피처
    permute는 "새 자리마다 옛날 몇 번 축을 가져올지"를 적는다.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling1D(nn.Module):
    """시계열 축 전체를 attention 가중 평균으로 요약한다.

    논문은 LSTM의 '마지막 hidden state'만 썼지만, lens는 모든 timestep을
    학습된 가중치로 합친다. "어느 날이 중요한지"를 모델이 스스로 정한다.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H]
        scores = self.attn(x)                   # [B, L, 1]  각 timestep의 점수

        # dim=1 = 시간축. 120일끼리 경쟁시켜 합이 1인 확률로 만든다.
        # (dim=2로 하면 축 길이가 1이라 전부 1.0이 되는 조용한 버그)
        weights = torch.softmax(scores, dim=1)  # [B, L, 1]

        # 시간축으로 가중합 → 시간축이 사라진다
        return (x * weights).sum(dim=1)         # [B, H]


class CNNLSTM(nn.Module):
    def __init__(
        self,
        n_features: int = 36,
        cnn_channels: int = 64,
        lstm_hidden: int = 128,
        n_layers: int = 2,
        horizon: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # 2배씩 키우면 receptive field가 지수적으로 커진다.
        # RF = 1 + 2×(1+2+4+8) = 31일
        self.dilations = (1, 2, 4, 8)

        conv_layers = []
        norm_layers = []
        in_channels = n_features
        for dilation in self.dilations:
            # padding=dilation 인 이유:
            #   conv는 양 끝에서 (k-1)×d = 2d 칸을 깎아먹는다.
            #   padding=d 면 양쪽 d칸씩 총 2d칸을 덧대므로 길이가 그대로 유지된다.
            conv_layers.append(
                nn.Conv1d(in_channels, cnn_channels, kernel_size=3,
                          padding=dilation, dilation=dilation)
            )
            norm_layers.append(nn.LayerNorm(cnn_channels))
            in_channels = cnn_channels

        self.conv_layers = nn.ModuleList(conv_layers)
        self.conv_norms = nn.ModuleList(norm_layers)
        # 입력을 conv 출력 채널 수로 맞춰주는 1x1 conv (residual 용)
        self.conv_residual_proj = nn.Conv1d(n_features, cnn_channels, kernel_size=1)
        self.conv_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            cnn_channels,
            lstm_hidden,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden)
        self.attn_pool = AttentionPooling1D(lstm_hidden)
        self.output_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden, horizon)

    @property
    def receptive_field(self) -> int:
        # RF = 1 + Σ (k-1)×d_i,  k=3 이므로 (k-1)=2
        return 1 + 2 * sum(self.dilations)

    def _forward_conv_stack(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F] = [B, 120, 36]

        # Conv1d는 [B, 채널, 길이]를 기대한다. 시간(1번)과 피처(2번)를 맞바꾼다.
        # 이걸 빠뜨리면 필터가 시간축이 아니라 피처축을 훑는다 → 조용히 망가짐
        channel_first = x.permute(0, 2, 1)      # [B, 120, 36] → [B, 36, 120]

        residual = self.conv_residual_proj(channel_first)
        hidden = channel_first
        for conv, norm in zip(self.conv_layers, self.conv_norms):
            hidden = conv(hidden)
            # LayerNorm은 마지막 축을 정규화하므로 잠깐 축을 바꿨다 되돌린다
            hidden = norm(hidden.permute(0, 2, 1)).permute(0, 2, 1)
            hidden = F.relu(hidden)
            hidden = self.conv_dropout(hidden)

        # residual 덧셈: LSTM의 C_t = f⊙C_{t-1} + i⊙C̃_t 와 같은 "덧셈으로 흘려보내기"
        return self.conv_dropout(hidden + residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self._forward_conv_stack(x)            # [B, 64, 120]

        # nn.LSTM(batch_first=True)는 [B, 길이, 채널]을 기대 → 다시 맞바꾼다
        sequence_hidden = hidden.permute(0, 2, 1)       # [B, 64, 120] → [B, 120, 64]

        lstm_out, _ = self.lstm(sequence_hidden)        # [B, 120, 128]
        lstm_out = self.lstm_norm(lstm_out)
        pooled = self.attn_pool(lstm_out)               # [B, 128]  시간축 소멸
        pooled = self.output_dropout(pooled)
        return self.head(pooled)                        # [B, 5]
