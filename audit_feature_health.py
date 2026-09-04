# -*- coding: utf-8 -*-
"""特徴量が実際に働いているかを検査する（2026-09-04）

きっかけ
  コース脚質バイアスが「常に0」の壊れた列だった。
  `df["馬場"]` を見ていたが、その列は存在しなかった。
  直したら BT の回収率が 108.1% → 122.5% に変わった。

  **何年も気づかなかったのは、特徴量が働いているかを見る検査が無かったから。**
  エラーは出ない。数字も出る。ただ静かに無意味な列になっていた。

同じ形で壊れている列を探す。壊れた列は次のどれかに当たる。
  ① 値が1種類しかない（定数になっている）
  ② 種類が極端に少ない
  ③ ほぼ全部欠損
  ④ モデルの重要度がほぼ0（使われていない＝壊れている可能性）

⚠ 重要度0＝壊れている、ではない。本当に効かない列もある。
  **人が見て判断する материал を出すだけ。**
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import pickle

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def log(m):
    print(m, flush=True)


def main():
    m = pickle.load(open(os.path.join(BASE_DIR, "model_resid.pkl"), "rb"))
    cols = m["use_cols"]
    models = m["models"]

    # 重要度（複数シードの平均）
    imp = np.zeros(len(cols))
    for mdl in models:
        imp += mdl.feature_importance(importance_type="gain")
    imp = imp / len(models)
    imp = imp / max(imp.sum(), 1e-9) * 100        # 割合(%)

    # 値の分布
    BATCH = 40
    stats = {}
    for i in range(0, len(cols), BATCH):
        part = cols[i:i + BATCH]
        acc_na = np.zeros(len(part))
        uniq = [set() for _ in part]
        n = 0
        for ch in pd.read_csv(os.path.join(BASE_DIR, "race_features.csv"),
                              usecols=part, chunksize=100000, low_memory=True):
            ch = ch.reindex(columns=part)
            acc_na += ch.isna().sum().values
            for j, c in enumerate(part):
                if len(uniq[j]) < 50:
                    uniq[j] |= set(pd.to_numeric(ch[c], errors="coerce")
                                   .dropna().unique()[:50])
            n += len(ch)
        for j, c in enumerate(part):
            stats[c] = {"欠損": acc_na[j] / n * 100, "種類": len(uniq[j])}

    R = pd.DataFrame([{"列": c, "重要度%": imp[k],
                       "欠損%": stats[c]["欠損"], "値の種類": stats[c]["種類"]}
                      for k, c in enumerate(cols)])
    R = R.sort_values("重要度%")

    log(f"  特徴量 {len(R)}列 / 学習データ {n:,}行\n")

    log("  === ① 値が1〜2種類しかない列（定数化を疑う） ===")
    x = R[R["値の種類"] <= 2].sort_values("重要度%", ascending=False)
    if len(x):
        for _, r in x.iterrows():
            log("    %-32s 種類%2d  欠損%5.1f%%  重要度%6.3f%%"
                % (r["列"][:32], r["値の種類"], r["欠損%"], r["重要度%"]))
    else:
        log("    無し")

    log("")
    log("  === ② ほぼ全部欠損（80%超） ===")
    x = R[R["欠損%"] > 80].sort_values("欠損%", ascending=False)
    if len(x):
        for _, r in x.head(10).iterrows():
            log("    %-32s 欠損%5.1f%%  重要度%6.3f%%"
                % (r["列"][:32], r["欠損%"], r["重要度%"]))
    else:
        log("    無し")

    log("")
    log("  === ③ 重要度がほぼ0の列（下位20） ===")
    log("  ⚠ 効かない列と壊れた列の両方が混じる。名前から見当をつける")
    for _, r in R.head(20).iterrows():
        log("    %-32s 重要度%7.4f%%  欠損%5.1f%%  種類%4d"
            % (r["列"][:32], r["重要度%"], r["欠損%"], r["値の種類"]))

    log("")
    log("  === ④ 重要度の上位10（参考・正常に効いている列） ===")
    for _, r in R.tail(10).iloc[::-1].iterrows():
        log("    %-32s 重要度%7.4f%%  欠損%5.1f%%"
            % (r["列"][:32], r["重要度%"], r["欠損%"]))

    log("")
    log("  === まとめ ===")
    log(f"    重要度0.01%未満の列  {(R['重要度%'] < 0.01).sum()} / {len(R)}")
    log(f"    重要度0.05%未満の列  {(R['重要度%'] < 0.05).sum()} / {len(R)}")
    log(f"    上位20列で重要度の  {R.tail(20)['重要度%'].sum():.1f}% を占める")

    R.to_csv(os.path.join(BASE_DIR, "feature_health.csv"),
             index=False, encoding="utf-8-sig")
    log("\n  feature_health.csv に保存")


if __name__ == "__main__":
    main()
