# -*- coding: utf-8 -*-
"""順位の確率を Harville（Plackett-Luce）で正しく組む（2026-09-04）

なぜ要るか
  ワイド（2頭とも3着以内）の確率を、独立の積 p3(A)×p3(B) で近似していた。
  実測すると

      独立の積の平均  46.9%
      実際の的中率    30.5%
      実際 ÷ 積 = 0.650

  **1.5倍に見積もっていた。**期待値に使えば必ず買いすぎる。
  馬は同じ枠を奪い合うので独立ではない。順位の確率モデルで組む必要がある。

Harville の考え方
  勝つ確率 π_i を「強さ」とみなし、1着が決まったら残りで2着を決める。

      P(i→j→k) = π_i × π_j/(1-π_i) × π_k/(1-π_i-π_j)

  これを必要な順序すべてについて足す。

提供するもの
  top3_prob(pi)          各馬が3着以内に入る確率
  pair_top3_prob(pi,a,b) 2頭とも3着以内に入る確率
  pair_top2_prob(pi,a,b) 2頭で1-2着を占める確率（馬連）

⚠ π は「その馬が勝つ確率」で、レース内で合計1になっていること。
"""
import numpy as np

EPS = 1e-12


def _norm(pi):
    pi = np.asarray(pi, dtype=float)
    pi = np.clip(pi, EPS, None)
    return pi / pi.sum()


def top3_prob(pi):
    """各馬が3着以内に入る確率。O(n^2) で厳密に計算する。

    P(i が3着以内) = P(1着) + P(2着) + P(3着)
      P(i が2着) = Σ_j≠i  π_j × π_i/(1-π_j)
      P(i が3着) = Σ_j≠i Σ_k≠i,j  π_j × π_k/(1-π_j) × π_i/(1-π_j-π_k)
    """
    pi = _norm(pi)
    n = len(pi)
    if n <= 3:
        return np.ones(n)
    p1 = pi.copy()

    # 2着: Σ_j π_j * π_i/(1-π_j)
    w = pi / np.clip(1.0 - pi, EPS, None)          # π_j/(1-π_j) ではなく後で使う形
    # P(i2) = Σ_{j≠i} π_j * π_i / (1-π_j)
    coef = pi / np.clip(1.0 - pi, EPS, None)       # π_j/(1-π_j)
    s_all = coef.sum()
    p2 = pi * (s_all - coef)                       # j≠i の分だけ

    # 3着: Σ_{j≠i} Σ_{k≠i,j} π_j π_k/(1-π_j) × π_i/(1-π_j-π_k)
    #   n は最大18程度なので二重ループで足りる
    p3 = np.zeros(n)
    for j in range(n):
        rem_j = 1.0 - pi[j]
        if rem_j <= EPS:
            continue
        for k in range(n):
            if k == j:
                continue
            rem_jk = rem_j - pi[k]
            if rem_jk <= EPS:
                continue
            base = pi[j] * pi[k] / rem_j
            contrib = base * pi / rem_jk           # 全 i について一括
            contrib[j] = 0.0
            contrib[k] = 0.0
            p3 += contrib
    return np.clip(p1 + p2 + p3, 0.0, 1.0)


def _order3(pi, a, b):
    """a と b が両方3着以内に入る確率。3枠のうち2つを a,b が占める場合を足す。"""
    pi = _norm(pi)
    n = len(pi)
    if n <= 3:
        return 1.0
    idx = [i for i in range(n) if i not in (a, b)]
    total = 0.0
    # a,b が入る2つの枠の位置（3枠から2つ選ぶ順序つき）と、残り1枠の馬 c
    slots = [(0, 1), (0, 2), (1, 2)]
    for s1, s2 in slots:
        third = ({0, 1, 2} - {s1, s2}).pop()
        for (x, y) in ((a, b), (b, a)):
            for c in idx:
                order = [None, None, None]
                order[s1], order[s2], order[third] = x, y, c
                p, rem = 1.0, 1.0
                ok = True
                for h in order:
                    if rem <= EPS:
                        ok = False
                        break
                    p *= pi[h] / rem
                    rem -= pi[h]
                if ok:
                    total += p
    return min(total, 1.0)


def pair_top3_prob(pi, a, b):
    """2頭とも3着以内（ワイド）。"""
    return _order3(pi, a, b)


def pair_top2_prob(pi, a, b):
    """2頭で1-2着を占める（馬連）。順不同。"""
    pi = _norm(pi)
    pa, pb = pi[a], pi[b]
    t = 0.0
    if 1.0 - pa > EPS:
        t += pa * pb / (1.0 - pa)
    if 1.0 - pb > EPS:
        t += pb * pa / (1.0 - pb)
    return min(t, 1.0)
