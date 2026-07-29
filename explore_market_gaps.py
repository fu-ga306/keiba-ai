# -*- coding: utf-8 -*-
"""③市場の情報が薄くなる場面を探す ＋ ①レース性質予測 ＋ ②時間軸(次走)。

これまでの結論: モデルは市場と互角だが超えられない。ならば「市場が正確でなくなる
場面」を特定するのが論理的な次の一手。

③市場の効率が落ちる候補:
   新馬戦(過去成績なし) / 開催初日(馬場読み不能) / 少頭数 / 長期休養明け /
   初コース・初距離 / 3歳未勝利の終盤 / 特殊条件
   → 判定は「オッズの織り込み精度」= 実勝率 vs 市場想定勝率 の乖離、
     および全馬購入時の単勝ROI（控除率20%＝理論80%からどれだけズレるか）。
①レース性質: 荒れる/堅いを事前に判別し、レースごとに買う券種を変えられるか。
②時間軸: 今走は負けても次走で狙える馬を見つけられるか（次走成績の予測）。
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def log(m):
    print(m, flush=True)


def load():
    d = pd.read_csv("race_features.csv", dtype={"race_id": str})
    for c in ["着順_num", "単勝オッズ", "人気", "出走頭数", "クラス_num", "距離",
              "is_turf", "馬番", "過去出走数", "前走間隔", "初コースフラグ",
              "距離変化", "馬場状態_num"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["着順_num", "単勝オッズ", "人気"])
    d["win"] = (d["着順_num"] == 1).astype(float)
    d["fuku"] = (d["着順_num"] <= 3).astype(float)
    d["bn"] = d["馬番"].astype("Int64").map(lambda x: f"{int(x):02d}" if pd.notna(x) else None)
    d["raw"] = 1 / d["単勝オッズ"]
    d["q"] = d["raw"] / d.groupby("race_id")["raw"].transform("sum")
    d["年"] = d["race_id"].str[:4].astype(int)
    d["日目"] = d["race_id"].str[8:10].astype(int)
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    tan = {(r.race_id, r.組み合わせ): r.払戻金
           for r in jv[jv["券種"] == "単勝"].itertuples()}
    return d, tan


def roi(s, tan):
    if not len(s):
        return float("nan")
    return sum(tan.get((r.race_id, r.bn), 0) for r in s.itertuples()) / (len(s) * 100) * 100


def part3(d, tan):
    log("=" * 92)
    log("【③市場の効率が落ちる場面】全馬購入の単勝ROI（理論値80%＝控除率20%）")
    log("   80%より高い＝市場が甘い（歪みが取り返せる） / 低い＝市場が辛い")
    log("=" * 92)
    log(f"  {'場面':<28}{'n':>8}{'全馬ROI':>9}{'1番人気ROI':>11}{'1人気複勝率':>11}")

    def seg(mask, label):
        s = d[mask]
        if len(s) < 2000:
            return
        f = s[s["人気"] == 1]
        log(f"  {label:<28}{len(s):8,}{roi(s, tan):8.1f}%{roi(f, tan):10.1f}%"
            f"{f['fuku'].mean()*100:10.1f}%")

    seg(pd.Series(True, index=d.index), "全体(基準)")
    seg(d["クラス_num"] == 0, "新馬戦(過去成績なし)")
    seg(d["過去出走数"] <= 1, "キャリア1走以下")
    seg(d["過去出走数"] >= 10, "キャリア10走以上")
    seg(d["日目"] <= 2, "開催1-2日目(馬場読み薄)")
    seg(d["日目"] >= 7, "開催7日目以降")
    seg(d["出走頭数"] <= 10, "少頭数(≤10)")
    seg(d["出走頭数"] >= 16, "多頭数(16+)")
    seg(d["前走間隔"] > 25, "長期休養明け(>25週)")
    seg(d["前走間隔"] <= 3, "連闘(≤3週)")
    if "初コースフラグ" in d.columns:
        seg(d["初コースフラグ"] == 1, "初コース")
    seg(d["距離変化"].abs() >= 400, "距離大幅変更(±400m)")
    seg(d["馬場状態_num"] >= 2, "道悪")
    seg(d["クラス_num"] >= 5, "OP・重賞")
    seg(d["is_turf"] == 0, "ダート")


def part1(d):
    log("\n" + "=" * 92)
    log("【①レース性質の予測】荒れる/堅いを事前に判別できるか")
    log("=" * 92)
    r = d.groupby("race_id").agg(
        頭数=("馬番", "size"), 勝ちオッズ=("単勝オッズ", lambda s: np.nan),
        q_top=("q", "max"), q_std=("q", "std"), cls=("クラス_num", "first"),
        turf=("is_turf", "first"), dist=("距離", "first"), day=("日目", "first"))
    # 同着があるとrace_idが重複するので、勝ち馬の最小オッズを代表値にする
    w = d[d["着順_num"] == 1].groupby("race_id")["単勝オッズ"].min()
    r["勝ちオッズ"] = r.index.map(w)
    r = r.dropna(subset=["勝ちオッズ"])
    r["荒れ"] = (r["勝ちオッズ"] >= 10).astype(float)
    log(f"  対象 {len(r):,}レース / 荒れた(勝ち馬10倍以上)率 {r['荒れ'].mean()*100:.1f}%")
    log(f"  {'指標':<24}{'AUC':>8}  ※0.5=判別不能")
    for c, nm in [("q_top", "1番人気への支持率"), ("q_std", "オッズ分布のばらつき"),
                  ("頭数", "出走頭数")]:
        s = r.dropna(subset=[c])
        log(f"  {nm:<24}{roc_auc_score(s['荒れ'], -s[c]):8.4f}")
    log("  → 市場のオッズ分布から『荒れやすさ』は読めるが、それは市場も知っている。")
    log("    実際に荒れた時に得をするには『どの穴馬が来るか』が要り、そこは前回까지で否定済み。")


def part2(d):
    log("\n" + "=" * 92)
    log("【②時間軸: 次走で狙えるか】今走の敗因が明確な馬は次走で走るか")
    log("=" * 92)
    import features as F
    s = d[["race_id", "馬名", "着順_num", "単勝オッズ", "人気", "出走頭数"]].copy()
    s = F.sort_by_horse_time(F.attach_race_date(s))
    s["次走着順"] = s.groupby("馬名")["着順_num"].shift(-1)
    s["次走人気"] = s.groupby("馬名")["人気"].shift(-1)
    s["次走複勝"] = (s["次走着順"] <= 3).astype(float)
    s = s.dropna(subset=["次走着順", "次走人気"])
    s = s[s["race_id"].str[:4] == "2025"]
    base = s["次走複勝"].mean() * 100
    log(f"  対象 {len(s):,}走 / 次走の複勝率(全体) {base:.1f}%")
    log(f"  {'今走の状況':<30}{'n':>8}{'次走複勝率':>11}{'次走平均人気':>12}")

    def seg(mask, label):
        x = s[mask]
        if len(x) < 500:
            return
        log(f"  {label:<30}{len(x):8,}{x['次走複勝'].mean()*100:10.1f}%"
            f"{x['次走人気'].mean():11.1f}")

    seg(s["着順_num"] == 1, "今走1着")
    seg(s["着順_num"].between(2, 3), "今走2-3着")
    seg(s["着順_num"].between(4, 6), "今走4-6着")
    seg(s["着順_num"] >= 10, "今走10着以下")
    seg((s["人気"] <= 3) & (s["着順_num"] >= 8), "人気で凡走(3人気以内→8着以下)")
    seg((s["人気"] >= 8) & (s["着順_num"] <= 5), "人気薄で健闘(8人気以下→5着以内)")
    log("  → 『人気で凡走』が次走も走らないなら市場は正しく割り引いている。")
    log("    次走人気が下がるのに複勝率が保たれる層があれば、そこが狙い目になる。")


def main():
    d, tan = load()
    log(f"データ {len(d):,}頭 / {d['race_id'].nunique():,}レース "
        f"({d['年'].min()}-{d['年'].max()})\n")
    part3(d, tan)
    part1(d)
    part2(d)


if __name__ == "__main__":
    main()
