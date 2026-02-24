# Paper Review: Attention Is All You Need

> Vaswani et al., NIPS 2017  
> 발표자: 김지형 (세종대학교)

---

## 논문 개요

| 항목 | 내용 |
|------|------|
| 제목 | Attention Is All You Need |
| 저자 | Vaswani et al. |
| 학회 | NIPS 2017 |
| 핵심 아이디어 | Recurrence 없이 Attention만으로 Seq2Seq 수행 |

---

## Introduction: 왜 Transformer인가?

기존 NLP는 RNN/LSTM 기반 Seq2Seq 모델이 지배적이었으나 두 가지 근본적인 한계가 있었습니다.

- **순차 처리(Sequential computation)** → 병렬화 불가능, GPU 자원 낭비
- **장기 의존성 문제** → 긴 문장에서 초기 정보 손실 및 Gradient Vanishing

> "Recurrence 없이 Attention만으로 충분하다"

### Sequence Model 발전 흐름

```
~2014              2015                    2017
RNN/LSTM/GRU  →  Seq2Seq + Attention  →  Transformer
(Sequential)     (Bahdanau et al.)       (Pure Attention)
```

---

## Background: 기존 모델의 한계

| Model | 시간 복잡도 | 경로 길이 | 문제점 |
|-------|------------|----------|--------|
| RNN | O(n) | O(n) | 순차 처리 병목, 장기 의존성 취약 |
| CNN | O(log n) | O(log n) | Receptive Field 제한 |
| **Transformer** | **O(1)** | **O(1)** | 해결 |

---

## Motivation & Goal

-  **완전한 병렬화**: 모든 토큰을 동시에 처리하여 GPU 자원 극대화  
-  **상수 시간 연결**: 모든 단어 간 직접 연결(Direct Access)로 경로 최소화  
-  **장기 의존성 해결**: 문장 길이에 상관없이 정보 손실 없는 학습

---

## Model Architecture

### 전체 구조

- **Encoder-Decoder Stack**: 각각 N=6개의 동일한 레이어
- **Multi-Head Attention**: 글로벌 의존성 포착
- **Position-wise FFN**: 각 위치에서 독립적인 비선형 변환
- **Residual & LayerNorm**: 각 서브레이어에 Add & Norm 적용
- **Positional Encoding**: Sine/Cosine 함수로 위치 정보 주입

---

## Core Mechanism: Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| 기호 | 의미 |
|------|------|
| Q (Query) | $W^Q \times X$ — 찾는 대상 |
| K (Key) | $W^K \times X$ — 매칭 기준 |
| V (Value) | $W^V \times X$ — 정보 내용 |
| $\sqrt{d_k}$ | 스케일링 — 기울기 안정화 |

**계산 흐름**: Q·Kᵀ 유사도 계산 → √dk로 스케일링 → Softmax → V 가중합

---

## Multi-Head Attention

단일 Attention의 편향 위험을 완화하고, 다양한 관점의 정보를 병렬로 학습합니다.

- **H=8개의 Head**: 서로 다른 문맥적 특징을 동시에 포착
- **차원 분할**: d_model(512)를 h=8로 분할 → 각 Head는 d_k=64 차원에서 연산
- **Ensemble Effect**: 각 Head 출력을 Concat 후 Linear로 통합

---

## Positional Encoding

Recurrence가 없어 위치 정보를 별도로 주입해야 합니다.

$$PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

**Sinusoidal을 선택한 이유:**
- 상대적 위치를 선형 변환으로 표현 가능
- 각 차원마다 고유한 파장 주기 보유
- 학습 시퀀스보다 긴 입력도 처리 가능 (Extrapolation)

---

## Training Configuration

| 항목 | Base | Big |
|------|------|-----|
| Optimizer | Adam (β₁=0.9, ε=10⁻⁹) | 동일 |
| Dropout | P=0.1 | - |
| Label Smoothing | ε=0.1 | - |
| Hardware | 8× NVIDIA P100 | 8× NVIDIA P100 |
| 학습 시간 | 12시간 (100K steps) | 3.5일 (300K steps) |
| warmup_steps | 4,000 | - |

---

## Experimental Results

### Translation Performance (EN-DE)

| Model | BLEU | Training Cost |
|-------|------|---------------|
| Previous SOTA | 25.16 | - |
| Transformer Base | 27.3 | 12h (8GPU) |
| **Transformer Big ★** | **28.4** | 3.5d (8GPU) |

- EN-FR: **41.8 BLEU** (당시 SOTA)
- 기존 모델 대비 **1/4 ~ 1/10 비용**으로 달성

### Ablation Study

| 요소 | 결과 |
|------|------|
| Heads h=8 | 최적 성능 |
| h=1 또는 h=16 | -0.9 BLEU 저하 |
| Positional Encoding 제거 | 성능 대폭 하락 |
| Dropout | Overfitting 방지에 필수 |

---

## Strengths

1. **완전한 병렬 처리**: RNN O(n) → Transformer O(1)
2. **장기 의존성 해결**: 경로 길이 O(n) → O(1) 직접 연결
3. **범용성과 확장성**: NLP(번역, 요약, QA), Vision(ViT, CLIP) 등 광범위 적용

---

## Limitations & Solutions

| 한계 | 설명 | 해결책 |
|------|------|--------|
| Quadratic Complexity | 시퀀스 길이에 따라 O(n²) 연산 증가 | Efficient Attention (Linear) |
| Simple PE | 고정된 Sin/Cos 방식의 표현 한계 | RoPE, ALiBi 등 고급 PE |
| High Resource | 대규모 학습 비용 (BERT Base: 16 TPU × 4일) | Distillation, Quantization |

---

## Impact & Future

- **BERT, RoBERTa**: Encoder 기반, 양방향 문맥 이해
- **GPT-4, ChatGPT**: Decoder-only, 인간 수준 텍스트 생성
- **ViT, CLIP, DALL-E**: CNN 없이 이미지 처리 가능

> "RNN 없는 시퀀스 모델링" 표준 확립 → 현대 AI의 Foundation Architecture

---

## Files

| 파일 | 설명 |
|------|------|
| `attention_is_all_you_need.pptx` | 발표 슬라이드 |

---

## Reference

- Vaswani, A., et al. (2017). *Attention Is All You Need*. NIPS 2017.  
  [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
