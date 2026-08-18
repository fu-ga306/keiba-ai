# -*- coding: utf-8 -*-
"""学習用と本番用の血統スナップショットの違いが、判断をどれだけ変えるか（2026-08-18）

見つけた欠陥
  学習・検証は sire_stats_father_train.csv（≤2024・626頭）で特徴量を作る。
  本番予測は sire_stats_father.csv（全期間・708頭）を使う。
  つまり同じ馬に、学習時と予測時で違う血統値が入る。

    父_複勝率     平均差 0.0175（中央値0.196に対し9%）  最大差 0.385
    父_長距離勝率  平均差 0.0116（中央値0.025に対し46%） 最大差 0.333

  モデルは≤2024の値で木を作ったのに、本番では全期間の値を渡される。
  木の分岐が変わるので判断が狂う可能性がある。

この検証で見ること
  血統列だけを本番用の値に差し替え、残差モデルの出力がどれだけ変わるかを測る。

    gapの変化が小さい        → この不一致は実害が無い。現状維持でよい
    軸に選ぶ馬が頻繁に変わる  → 実害がある。本番も学習用に揃えるべき

  ⚠ 学習には一切触らない。入力を差し替えるだけなのでリークしない。
  ⚠ 全期間版は2025年の結果も含むので、本番の「予測時点」より情報が多い。
    つまりここで出る差は**影響の上限**。実際はもっと小さい。

実行: python diag_snapshot.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import resid_io

BLOOD = ["父系_今回距離適性", "母父系_今回距離適性", "父系_長距離勝率",
         "母父系_長距離勝率", "父系_芝ダ適性", "父系_複勝率"]
TEST_YEARS = ["2024", "2025"]
rng = np.random.default_rng(20260818)


def log(m):
    print(m, flush=True)


def band(d):
    if pd.isna(d):
        return None
    return "短距離" if d <= 1400 else ("中距離" if d <= 2000 else "長距離")


def blood_values(df, suffix):
    """指定のスナップショットで血統6列を作り直す。features.py と同じ引き方。"""
    f = pd.read_csv(f"sire_stats_father{suffix}.csv")
    b = pd.read_csv(f"sire_stats_bms{suffix}.csv")
    fd = f.set_index("父名").to_dict("index")
    bd = b.set_index("母父名").to_dict("index") if "母父名" in b.columns else {}

    def cm(d, col):
        return {k: v.get(col, np.nan) for k, v in d.items()}

    sire, bms = df["父馬"], df["母父馬"]
    dist = pd.to_numeric(df["距離"], errors="coerce")
    turf = pd.to_numeric(df["is_turf"], errors="coerce")
    bnd = dist.map(band)
    out = pd.DataFrame(index=df.index)
    out["父系_長距離勝率"] = sire.map(cm(fd, "父_長距離勝率"))
    out["母父系_長距離勝率"] = bms.map(cm(bd, "母父_長距離勝率"))
    out["父系_複勝率"] = sire.map(cm(fd, "父_複勝率"))
    out["父系_芝ダ適性"] = np.where(turf == 1, sire.map(cm(fd, "父_芝勝率")),
                              sire.map(cm(fd, "父_ダート勝率")))
    for col, d, pre in (("父系_今回距離適性", fd, "父_"),
                        ("母父系_今回距離適性", bd, "母父_")):
        v = pd.Series(np.nan, index=df.index)
        src = sire if pre == "父_" else bms
        for nm in ("短距離", "中距離", "長距離"):
            m = bnd == nm
            if m.any():
                v[m] = src[m].map(cm(d, pre + nm + "勝率"))
        out[col] = v
    return out


def main():
    log("血統列だけを差し替えて、残差モデルの出力の変化を測ります\n")
    hm = pd.read_csv("horse_master.csv")
    hm["horse_id"] = hm["horse_id"].astype(str).str.replace(".0", "", regex=False).str.strip()
    rc = pd.read_csv("race_data_clean.csv", usecols=["race_id", "馬名", "horse_id"],
                     dtype=str, low_memory=False)
    rc["race_id"] = rc["race_id"].str.replace(r"\.0$", "", regex=True)
    rc["horse_id"] = rc["horse_id"].astype(str).str.replace(".0", "", regex=False).str.strip()
    rc = rc.merge(hm[["horse_id", "父馬", "母父馬"]], on="horse_id", how="left")

    m = resid_io.load_model()
    cols = m["use_cols"]
    use = list(dict.fromkeys(["race_id", "馬名", "馬番", "単勝オッズ", "人気",
                              "着順_num", "is_turf", "距離"] + cols + BLOOD))
    D = pd.read_csv("race_features.csv", usecols=use, dtype={"race_id": str},
                    low_memory=False)
    D["race_id"] = D["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D["年"] = D["race_id"].str[:4]
    D = D[D["年"].isin(TEST_YEARS)].copy()
    D = D.merge(rc[["race_id", "馬名", "父馬", "母父馬"]].drop_duplicates(["race_id", "馬名"]),
                on=["race_id", "馬名"], how="left")
    D["odds"] = pd.to_numeric(D["単勝オッズ"], errors="coerce")
    D = D[D.odds > 0].copy().reset_index(drop=True)
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース（{TEST_YEARS}）")

    # 現状（学習用スナップショット）の予測
    d_tr = resid_io.predict_gap(m, D)
    # 本番用スナップショットに差し替えた予測
    D2 = D.copy()
    nb = blood_values(D2, "")
    changed = 0
    for c in BLOOD:
        if c in D2.columns:
            diff = (pd.to_numeric(D2[c], errors="coerce")
                    - pd.to_numeric(nb[c], errors="coerce")).abs()
            changed += int((diff > 1e-9).sum())
            D2[c] = nb[c].values
    # レース内偏差・順位の派生列も作り直す
    for c in BLOOD:
        for suf, fn in (("_R偏差", lambda s: (s - s.mean()) / (s.std() if s.std() else 1)),
                        ("_R順", lambda s: s.rank(pct=True, ascending=False))):
            col = c + suf
            if col in D2.columns:
                D2[col] = D2.groupby("race_id")[c].transform(fn)
    log(f"血統値が変わった延べ件数 {changed:,}\n")
    d_pr = resid_io.predict_gap(m, D2)

    g1 = pd.to_numeric(d_tr["gap"], errors="coerce")
    g2 = pd.to_numeric(d_pr["gap"], errors="coerce")
    log("=== gapの変化 ===")
    log(f"  平均絶対差 {(g2-g1).abs().mean():.4f}   中央値 {(g2-g1).abs().median():.4f}")
    log(f"  最大差     {(g2-g1).abs().max():.4f}")
    log(f"  相関       {g1.corr(g2):.5f}")

    log("\n=== 軸に選ぶ馬が変わるか ===")
    t = pd.DataFrame({"rid": D.race_id, "bn": D["馬番"], "g1": g1, "g2": g2})
    same = tot = both = 0
    for rid, x in t.groupby("rid", sort=False):
        a = x.loc[x.g1.idxmax(), "bn"]
        b = x.loc[x.g2.idxmax(), "bn"]
        tot += 1
        same += int(a == b)
        if x.g1.max() >= resid_io.AX_GAP or x.g2.max() >= resid_io.AX_GAP:
            both += 1
    log(f"  同じ馬が軸になる割合 {same/tot*100:.1f}%（{tot:,}レース）")
    log(f"  → {'○ 実害は小さい' if same/tot >= 0.95 else '⚠ 判断が変わる。揃えるべき'}")

    log("\n=== 買う/買わないの判定が変わるレース ===")
    a1 = t.groupby("rid").g1.max() >= resid_io.AX_GAP
    a2 = t.groupby("rid").g2.max() >= resid_io.AX_GAP
    flip = int((a1 != a2).sum())
    log(f"  {flip:,}/{len(a1):,}レース（{flip/len(a1)*100:.1f}%）で判定が反転")
    log(f"  買う: 学習用 {int(a1.sum()):,} → 本番用 {int(a2.sum()):,}")


if __name__ == "__main__":
    main()
