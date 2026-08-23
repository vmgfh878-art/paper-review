"""
shape 검증 스크립트.

    python cnn-lstm/code/run_shapes.py                 # fill_in.py 채점
    python cnn-lstm/code/run_shapes.py --mode solution # 정답본 확인

더미 텐서를 흘려서 각 단계 shape가 기대값과 맞는지 본다.
추상 개념을 텐서 모양으로 눈으로 확인하는 게 목적.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

B, L, FEAT = 4, 120, 36
HORIZON = 5
EXPECTED_RF = 31


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fill_in", "solution"], default="solution")
    args = parser.parse_args()

    if args.mode == "solution":
        from solution import CNNLSTM
    else:
        try:
            from fill_in import CNNLSTM
        except SyntaxError as exc:
            print(f"[!] fill_in.py에 아직 빈칸(____)이 남아 있습니다.\n    {exc}")
            return 1

    torch.manual_seed(0)
    model = CNNLSTM(n_features=FEAT, horizon=HORIZON)
    model.eval()

    x = torch.randn(B, L, FEAT)
    print(f"input          [B, L, F] : {tuple(x.shape)}")

    ok = True

    # 1) conv 스택
    with torch.no_grad():
        conv_out = model._forward_conv_stack(x)
    print(f"after conv     [B, C, L] : {tuple(conv_out.shape)}   expected {(B, 64, L)}")
    if tuple(conv_out.shape) != (B, 64, L):
        print("    [X] channels must be 64 and length must stay 120 (check padding)")
        ok = False

    # 2) receptive field
    rf = model.receptive_field
    print(f"receptive field          : {rf}   expected {EXPECTED_RF}")
    if rf != EXPECTED_RF:
        print("    [X] RF = 1 + sum((k-1)*d_i),  k=3, d=(1,2,4,8)")
        ok = False

    # 3) 최종 출력
    with torch.no_grad():
        out = model(x)
    print(f"output         [B, H]    : {tuple(out.shape)}   expected {(B, HORIZON)}")
    if tuple(out.shape) != (B, HORIZON):
        print("    [X] attention pooling must collapse the time axis")
        ok = False

    # 4) attention 가중치가 확률인지 (timestep 합 = 1)
    with torch.no_grad():
        seq = model._forward_conv_stack(x).permute(0, 2, 1)
        lstm_out, _ = model.lstm(seq)
        scores = model.attn_pool.attn(model.lstm_norm(lstm_out))
        w = torch.softmax(scores, dim=1)
    total = w.sum(dim=1).flatten()
    print(f"attn weights sum         : {total[0].item():.4f}   expected 1.0000")
    if not torch.allclose(total, torch.ones_like(total), atol=1e-4):
        print("    [X] softmax must be applied along the time axis (dim=1)")
        ok = False

    print()
    print("[O] all checks passed" if ok else "[X] some checks failed - see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
