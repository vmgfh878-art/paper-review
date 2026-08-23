# paper review

읽은 논문을 이해한 만큼 정리해서 적어두는 공간입니다.
시계열, LLM, RL 등 그때그때 관심 있는 분야를 읽습니다.

---

## 읽은 논문

### 시계열

| 논문 | 출처 | |
|---|---|---|
| CNN-LSTM | Lu et al., Complexity 2020 | [리뷰](cnn-lstm/review.md) |
| PatchTST | Nie et al., ICLR 2023 | [리뷰](patchtst/review.md) |

### LLM

| 논문 | 출처 | |
|---|---|---|
| Attention Is All You Need | Vaswani et al., NIPS 2017 | [발표자료](attention-is-all-you-need/) |
| Contrastive Decoding | Li et al., ACL 2023 | [리뷰](contrastive-decoding/review.md) |

<!-- 새 분야는 논문이 들어올 때 섹션을 추가합니다 -->

## 읽을 논문

- TiDE — Das et al., TMLR 2023
- TCN — Bai et al., 2018
- DAPO — Yu et al., 2025
- 포트폴리오 이론 계열

---

## 리뷰 틀

[TEMPLATE.md](TEMPLATE.md) 를 복사해서 씁니다.

```
메타 (분야 / 읽은 날 / 읽은 이유)
1. 한 줄 요약
2. 배경 — 이 논문이 나오기까지
3. 기존 방식의 한계
4. 핵심 아이디어
5. 구조 / 방법
6. 주요 구성 요소
7. 실험 결과 (+ Ablation)
8. 한계와 후속
9. 참고
```

선택 블록은 해당될 때만 넣습니다 — 배경 지식 정리, 수학, 코드, 내 프로젝트 연계, 남은 질문.

## 폴더 구성

```
논문이름/
  review.md      # 리뷰
  code/          # 구현을 따라간 경우에만
```

코드가 있는 리뷰에는 shape 검증 스크립트를 함께 둡니다.

```bash
python cnn-lstm/code/run_shapes.py
```
