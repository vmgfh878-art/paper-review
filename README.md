# paper review

시계열 딥러닝 논문을 매주 한 편씩 리뷰한다.
논문을 읽고 끝내지 않고, **수학 유도 → 코드 확인 → 내 프로젝트(lens)와의 대조**까지 한 번에 묶는 것이 목표.

## 진행 현황

| # | 논문 | 난이도 | lens 대응 | 상태 |
|---|---|---|---|---|
| 01 | [CNN-LSTM](01-cnn-lstm/review.md) (Lu et al., Complexity 2020) | ★☆☆ | `ai/models/cnn_lstm.py` | ✅ 완료 |
| 02 | TCNQuantile (분위수 예측) | ★★☆ | `ai/models/tcn_quantile.py` | 예정 |
| 03 | TiDE (Das et al., TMLR 2023) | ★★☆ | `ai/models/tide.py` | 예정 |
| 04 | [PatchTST](04-patchtst/review.md) (Nie et al., ICLR 2023) | ★★★ | `ai/models/patchtst.py` | 🔶 작성 중 |

> 별도: [attention-is-all-you-need](attention-is-all-you-need/)

## 리뷰 구성

각 논문 폴더는 다음을 가진다.

```
NN-이름/
  review.md        # 리뷰 본문
  code/
    solution.py    # 논문 핵심 골격만 남긴 학습용 구현 (lens 코드 기반)
    run_shapes.py  # 더미 텐서로 forward, shape 검증
```

`review.md` 목차:

1. 한 줄 요약
2. 본문 해석 노트 — 데이터 흐름과 구조
3. 수학 리뷰 — 핵심 수식 1~2개 유도 + 직관
4. 코드 리뷰 — 실제 구현에서 헷갈리는 지점
5. lens 연계 — 논문 ↔ 내 코드 대조, 논문에 없는 것
6. 참고 + 셀프 퀴즈

## 주간 진행 순서

| 단계 | 내용 |
|---|---|
| ① 1-pass | 초록·그림·결론만 훑고 한 줄 요약 |
| ② 본문 해석 | 데이터 흐름 다이어그램 + 개념 퀴즈 |
| ③ 수학 리뷰 | 핵심 수식 유도 + 퀴즈 |
| ④ 코드 | 주석 달린 구현 읽고 shape 실행 확인 |
| ⑤ lens 연계 | 논문 ↔ 내 코드 매핑 |
| ⑥ 저장 | 커밋 |

논문 읽기는 [three-pass method](https://github.com/parksb/papers-i-love/blob/main/how-to-read-a-paper.md)를 따른다.

## 실행

```bash
python 01-cnn-lstm/code/run_shapes.py --mode solution
```
