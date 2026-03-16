# PatchTST 논문 리뷰
> A Time Series is Worth 64 Words: Long-term Forecasting with Transformers  
> Nie et al., ICLR 2023  
> https://arxiv.org/abs/2211.14730

---

## 1. 한 줄 요약

시계열을 날짜 단위가 아니라 패치(구간) 단위로 잘라서 Transformer에 입력하면
장기 예측 성능이 크게 올라간다.

---

## 2. 문제 제기 - 기존 Transformer의 한계

기존 시계열 Transformer들은 이런 방식으로 데이터를 봤다.

```
입력: [1일], [2일], [3일], [4일], ..., [512일]
         ↓      ↓      ↓      ↓            ↓
      토큰1  토큰2  토큰3  토큰4  ...    토큰512
```

문제 1 - Semantic이 부족하다  
하루치 데이터 하나(종가, RSI 등 몇 개 숫자)는 의미가 너무 작다.  
단어 하나하나가 의미를 가지는 NLP와 달리, 하루 주가 하나는 맥락 없이는 해석하기 어렵다.

문제 2 - 연산량이 너무 많다  
Transformer의 Attention 연산은 토큰 수의 제곱에 비례한다.  
512일치 데이터면 토큰 512개 → 512² = 262,144번 연산.  
입력이 길어질수록 메모리와 시간이 폭발적으로 늘어난다.

문제 3 - Look-ahead bias  
일부 모델들이 시계열 예측에서 미래 정보를 은연중에 참조하는 구조적 결함이 있었다.  
Time embedding이 미래 정보를 포함하는 경우가 해당된다.

---

## 3. 핵심 아이디어 - Patching

논문의 핵심 기여는 단순하고 강력하다.

```
기존 방식
[1일] [2일] [3일] [4일] [5일] [6일] [7일] [8일] ...
  ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓
토큰1 토큰2 토큰3 토큰4 토큰5 토큰6 토큰7 토큰8  (토큰 많음)


PatchTST 방식 (patch_len=4, stride=2)
[1~4일] [3~6일] [5~8일] ...
   ↓        ↓        ↓
 패치1    패치2    패치3   (토큰 줄어듦)
```

날짜 하나하나 보지 않고, 여러 날을 묶은 구간(패치) 하나를 토큰 하나로 본다.

효과 1 - 의미 있는 토큰  
16일치 데이터를 묶으면 "이 기간은 상승 추세였다" 같은 맥락이 생긴다.

효과 2 - 연산량 감소  
토큰 수가 줄어드므로 Attention 연산량도 줄어든다.  
512일 → patch_len=16, stride=8 → 토큰 약 63개  
512² vs 63² → 연산량이 약 66배 감소.

효과 3 - 로컬 시맨틱 보존  
인접한 시점들을 함께 묶으므로 단기 패턴 정보가 패치 안에 보존된다.

---

## 4. 모델 구조

```
입력 [Batch, Seq_Len, Features]
  ↓
RevIN (정규화)
  ↓
Channel Independence (피처 분리: [B, S, F] → [B*F, S])
  ↓
Patching (구간으로 자르기: [B*F, S] → [B*F, N, P])
  ↓
Patch Embedding + Positional Embedding
  ↓
Transformer Encoder (L번 반복)
  ↓
Flatten + MLP Head
  ↓
출력 [Batch, 1]
```

Transformer Encoder를 거친 후 각 패치마다 D차원 벡터가 남는다.
이 정보를 전부 하나의 확률값으로 압축하는 과정이 Flatten + MLP Head다.

```
Flatten:  [B, C, N, D] → [B, C, N*D]
# 패치 14개 × d_model 128 = 1792개 숫자를 한 줄로 펼치기

MLP Head: [B, C, N*D] → [B, 1]
# Linear 레이어 몇 개를 통과해서 숫자 하나로 압축
# 이 값이 최종 출력 "상승 확률 0.72" 같은 값
```

Flatten은 여러 차원의 정보를 한 줄로 세우는 것,
MLP Head는 그걸 받아서 최종 답 하나를 내는 것이다.

---

## 5. 주요 구성 요소 설명

### 5-1. RevIN (Reversible Instance Normalization)

시계열 데이터는 종목마다, 시기마다 가격 분포가 완전히 다르다.  
삼성전자는 50,000원대, 애플은 150달러대 → 같은 모델이 처리하기 어렵다.

RevIN은 이 문제를 해결한다.

```
입력 시 (norm):   평균 빼고 표준편차로 나눔  → 분포 통일
출력 시 (denorm): 원래 평균/표준편차로 복원 → 실제 값 복구
```

핵심은 정규화에 쓴 통계값(평균, 표준편차)을 기억해뒀다가 출력할 때 되돌린다는 것이다.  
일반적인 배치 정규화는 학습 전체 데이터의 통계를 쓰지만,  
RevIN은 샘플 하나하나의 통계를 쓰기 때문에 추론 시점의 분포 변화에 강하다.

### 5-2. Channel Independence (CI)

기존 방식은 모든 피처를 한꺼번에 처리했다 (Channel Mixing).  
PatchTST는 각 피처를 독립적으로 처리한다.

```
Channel Mixing:   [B, S, F] → Attention → 피처들이 섞임
Channel Indep:    [B, S, F] → [B*F, S] → 각 피처 따로 처리
```

RSI는 RSI끼리, MACD는 MACD끼리의 시간적 패턴을 독립적으로 학습한다.  
피처 간 간섭이 줄어들어 학습이 안정적이다.

### 5-3. Patching

```python
# PyTorch unfold로 구현
# [B*F, S] → [B*F, N, patch_len]
x = x.unfold(dimension=1, size=patch_len, step=stride)

# 패치 수 계산
N = (seq_len - patch_len) / stride + 1
# 예: seq_len=120, patch_len=16, stride=8
# N = (120 - 16) / 8 + 1 = 14개 패치
```

stride < patch_len 이면 패치가 겹친다(overlap).  
겹치는 구간이 두 패치 모두에 들어가므로 정보 손실이 줄어든다.

---

## 6. 실험 결과

논문에서 ETTh1, ETTm1, Weather, Traffic 등 8개 벤치마크 데이터셋에서 실험.

장기 예측(Long-term Forecasting)에서 기존 SOTA 대비 성능 비교

| 모델 | MSE (낮을수록 좋음) |
|---|---|
| FEDformer | 0.379 |
| Autoformer | 0.449 |
| Informer | 0.865 |
| PatchTST | 0.370 |

PatchTST가 당시 SOTA를 경신했고, 특히 예측 horizon이 길수록 성능 차이가 커진다.  
중장기 예측에 특히 강하다는 것이 핵심.

---

## 7. 코드 분석 - 논문 공식 구현

공식 구현: https://github.com/yuqinie98/PatchTST

---

**RevIN.py**

```python
def _get_statistics(self, x):
    self.mean = torch.mean(x, ...).detach()
    self.stdev = torch.sqrt(torch.var(x, ...) + self.eps).detach()

def _normalize(self, x):
    x = (x - self.mean) / self.stdev
    x = x * self.affine_weight + self.affine_bias

def _denormalize(self, x):
    x = (x - self.affine_bias) / self.affine_weight
    x = x * self.stdev + self.mean
```

`.detach()` 는 "이 값은 gradient 계산에 포함시키지 않겠다"는 선언이다.  
모델이 학습할 때 역전파(backpropagation)가 일어나는데, mean과 stdev는 단순히 "현재 입력의 통계값"이지 학습 파라미터가 아니다. 역전파가 여기까지 타고 들어오면 의도치 않게 정규화 통계가 gradient에 영향을 주게 된다. `.detach()`로 이걸 차단한다.

`affine_weight`(γ)와 `affine_bias`(β)는 학습 가능한 파라미터다.  
정규화 후 x가 모든 종목에 대해 똑같은 분포(평균 0, 표준편차 1)가 되는데, 이러면 종목마다의 고유한 특성이 사라진다. γ와 β를 두면 "이 피처는 정규화를 좀 약하게", "이 피처는 좀 강하게" 같은 걸 모델이 스스로 학습한다. 즉 정규화의 강도 자체를 학습하는 파라미터다.

denormalize에서는 normalize의 역순으로 γ, β를 먼저 되돌리고, 그 다음 저장해둔 mean과 stdev로 원래 스케일을 복원한다.

---

**PatchTST_backbone.py**

```python
# Patching - 시계열을 구간으로 자르기
z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
# [B, C, S] → [B, C, 패치수, 패치길이]

# Patch Embedding
z = self.W_P(z)  # Linear(patch_len, d_model)
# [B, C, N, patch_len] → [B, C, N, d_model]

# Channel Independence - 배치와 채널을 합쳐서 독립 처리
u = torch.reshape(z, (z.shape[0]*z.shape[1], z.shape[2], z.shape[3]))
# [B, C, N, D] → [B*C, N, D]

# Positional Embedding (학습 가능한 파라미터, sin/cos 아님)
u = self.dropout(u + self.W_pos)

# Transformer Encoder
z = self.encoder(u)

# 채널 복원 후 예측
z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
z = self.flatten(z)
z = self.head(z)
```

`unfold` 한 줄이 Patching 전부다.  
`reshape` 으로 배치와 채널을 합치는 것이 Channel Independence 구현의 전부다.

---

## 8. 참고

- 논문 원문: https://arxiv.org/abs/2211.14730
- 공식 구현: https://github.com/yuqinie98/PatchTST
