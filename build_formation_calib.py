"""
build_formation_calib.py
─────────────────────────
推奨馬の組み合わせ的中率（フォーメーション）を実績ベースに補正する
isotonicキャリブレーターを、model_result.csv（2025年バックテスト結果）から生成する。

生のPlackett-Luce確率は組み合わせ的中を+13〜24pt過大評価するため、
過去実績に合わせて補正する。model.py 再学習後に実行して再生成すること。

使い方:
    python build_formation_calib.py

入力:  model_result.csv（勝ち確率・予測順位・着順_num）
出力:  formation_calibrators.pkl（keiba_predict.py が読み込む）

※ ネット接続不要（ローカルCSVのみ使用）。
"""
import os
import pickle
import numpy as np
import pandas as pd
from itertools import permutations
from sklearn.isotonic import IsotonicRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def formation_probs(win_probs, idx3, idx5):
    """勝ち確率(Plackett-Luce)から推奨馬集合の組み合わせ確率を計算（生確率）。"""
    p = np.array(win_probs, dtype=float)
    p = np.clip(np.nan_to_num(p, nan=0.0), 1e-12, None)
    p = p / p.sum()
    n = len(p)
    S3, S5 = set(idx3), set(idx5)

    def top2_in(S):
        tot = 0.0
        for i in S:
            for j in S:
                if i == j:
                    continue
                d = 1 - p[i]
                if d > 1e-12:
                    tot += p[i] * p[j] / d
        return min(tot, 1.0)

    a2 = 0.0
    b2 = 0.0
    for i, j, k in permutations(range(n), 3):
        d1 = 1 - p[i]
        d2 = 1 - p[i] - p[j]
        if d1 <= 1e-12 or d2 <= 1e-12:
            continue
        pr = p[i] * (p[j] / d1) * (p[k] / d2)
        t3 = {i, j, k}
        if len(t3 & S3) >= 2:
            a2 += pr
        if t3 <= S5:
            b2 += pr
    return {"s3_2ren": top2_in(S3), "s3_2fuku": a2,
            "s5_2ren": top2_in(S5), "s5_3fuku": b2}


def main():
    path = os.path.join(BASE_DIR, "model_result.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.dropna(subset=["勝ち確率", "予測順位", "着順_num"])

    keys = ["s3_2ren", "s3_2fuku", "s5_2ren", "s5_3fuku"]
    pred = {k: [] for k in keys}
    hit = {k: [] for k in keys}

    for rid, g in df.groupby("race_id"):
        g = g.sort_values("予測順位").reset_index(drop=True)
        if len(g) < 6:
            continue
        wp = g["勝ち確率"].values
        idx3 = g.index[g["予測順位"] <= 3].tolist()
        idx5 = g.index[g["予測順位"] <= 5].tolist()
        if len(idx3) < 3 or len(idx5) < 5:
            continue
        r = formation_probs(wp, idx3, idx5)
        for k in keys:
            pred[k].append(r[k])
        s3 = g.loc[idx3]
        s5 = g.loc[idx5]
        hit["s3_2ren"].append(1 if (s3["着順_num"] <= 2).sum() >= 2 else 0)
        hit["s3_2fuku"].append(1 if (s3["着順_num"] <= 3).sum() >= 2 else 0)
        hit["s5_2ren"].append(1 if (s5["着順_num"] <= 2).sum() >= 2 else 0)
        hit["s5_3fuku"].append(1 if (s5["着順_num"] <= 3).sum() >= 3 else 0)

    labels = {"s3_2ren": "推奨3頭中2頭連対", "s3_2fuku": "推奨3頭中2頭複勝",
              "s5_2ren": "推奨5頭中2頭連対", "s5_3fuku": "推奨5頭中3頭複勝"}
    calibrators = {}
    print(f"検証レース数: {len(pred['s3_2ren'])}")
    print(f"{'指標':<18}{'生予測':>8}{'補正後':>8}{'実測':>8}")
    print("-" * 44)
    for k in keys:
        P = np.array(pred[k])
        H = np.array(hit[k])
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(P, H)
        calibrators[k] = iso
        cal = iso.predict(P)
        print(f"{labels[k]:<18}{P.mean()*100:>7.1f}%{cal.mean()*100:>7.1f}%{H.mean()*100:>7.1f}%")

    out = os.path.join(BASE_DIR, "formation_calibrators.pkl")
    with open(out, "wb") as f:
        pickle.dump(calibrators, f)
    print(f"\n補正モデル保存 → {out}")


if __name__ == "__main__":
    main()
