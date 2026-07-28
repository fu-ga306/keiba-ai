# -*- coding: utf-8 -*-
"""現行仕様(判定マトリクス+複妙軸+MF相手+予算配分)を2025全レースに適用し、
JV実払戻(jv_payouts.csv)で金額加重の実収支を出す忠実バックテスト。
本番の _race_bet_plan / _build_bet_rows / 資金設定 をそのまま呼ぶ。

印の再現(keiba_predict本体と同一):
  ◎=1番人気(オッズ≤2.0 or OP)else 主モデルplace3-1位 / ○▲△=place3順の残り上位3
  妙味軸=MF勝率1位(≠◎のとき) / 複妙=MF複勝1位 / ×は現行メニュー未使用のため省略
"""
import warnings; warnings.filterwarnings("ignore")
import os
import pandas as pd, numpy as np, collections
import keiba_predict as kp

UNORDERED = {"馬連", "ワイド", "3連複", "枠連"}
TEST_YEAR = os.environ.get("KEIBA_TEST_YEAR", "2025")   # 多年度検証用


def norm(kind, combo):
    p = str(combo).split("-")
    if kind in UNORDERED:
        p = sorted(p)
    return "-".join(p)


def load():
    p3 = pd.read_csv("model_result_place3.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "予測順位"]].rename(columns={"予測順位": "place3順"})
    mf = pd.read_csv("model_mf_result.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "MF勝率", "MF勝率順位", "MF複勝順位"]]
    rf = pd.read_csv("race_features.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "馬番", "人気", "単勝オッズ", "クラス_num", "着順_num"]]
    df = rf.merge(p3, on=["race_id", "馬名"], how="inner").merge(mf, on=["race_id", "馬名"], how="inner")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv = jv[jv["race_id"].str.startswith(TEST_YEAR)]
    pay = {(r.race_id, r.券種, r.組み合わせ): int(r.払戻金) for r in jv.itertuples()}

    # 期間分割（KEIBA_HALF=1/2: 実開催日で前半/後半のみ評価・頑健性確認用）
    hf = os.environ.get("KEIBA_HALF")
    if hf in ("1", "2"):
        dates = pd.read_csv("race_dates.csv", dtype={"kaisai_key": str})
        dmap = dict(zip(dates["kaisai_key"], pd.to_datetime(dates["date"])))
        dts = df["race_id"].str[:10].map(dmap)
        mid = dts.quantile(0.5)
        df = df[dts <= mid] if hf == "1" else df[dts > mid]
        print(f"[期間分割] {'前半' if hf=='1' else '後半'}のみ: {df['race_id'].nunique()}レース")

    # 前走間隔フィルタ（KEIBA_INTERVAL_FILTER=1で有効・2026-07-27検証）
    #   MFのエッジは通常ローテ(4-25週)でのみ成立し、連闘(≤3週)/長期休養明け(>25週)では
    #   単勝ROIが86%/85%と100%を割る（市場では同傾向が出ない＝MF固有の弱点。
    #   人気帯で層別しても2分割しても再現）。軸がその区間なら見送る。
    if os.environ.get("KEIBA_INTERVAL_FILTER") == "1":
        rf = pd.read_csv("race_features.csv", dtype={"race_id": str},
                         usecols=["race_id", "馬名", "前走間隔"])
        df = df.merge(rf, on=["race_id", "馬名"], how="left")
        iv = pd.to_numeric(df["前走間隔"], errors="coerce")
        df["_bad"] = ((iv <= 3) | (iv > 25)).fillna(False)
        axis = df[(df["MF勝率順位"] == 1) | (df["MF複勝順位"] == 1)]
        ng = set(axis[axis["_bad"]]["race_id"])
        before = df["race_id"].nunique()
        df = df[~df["race_id"].isin(ng)]
        print(f"[間隔フィルタ] 軸が連闘/半年+のレースを除外: "
              f"{before} → {df['race_id'].nunique()}レース")
    return df, pay


def build_pdf(g):
    """1レース分のper-horse frameに 印/妙味軸/MF勝ち確率 を付与して返す。"""
    g = g.copy()
    g["MF勝ち確率"] = g["MF勝率"]
    g["印"] = ""
    g["妙味軸"] = ""
    pop = pd.to_numeric(g["人気"], errors="coerce")
    cls = pd.to_numeric(g["クラス_num"].iloc[0], errors="coerce")
    fav_i = pop.idxmin() if pop.notna().any() else g.index[0]
    fav_odds = pd.to_numeric(g.loc[fav_i, "単勝オッズ"], errors="coerce")
    use_fav = (pd.notna(fav_odds) and float(fav_odds) <= 2.0) or (pd.notna(cls) and int(cls) >= 5)
    fuku = g.sort_values("place3順")   # place3昇順=複勝確率降順
    hon_i = fav_i if use_fav else fuku.index[0]
    g.at[hon_i, "印"] = "◎"
    for mk, idx in zip(("○", "▲", "△"), [i for i in fuku.index if i != hon_i][:3]):
        g.at[idx, "印"] = mk
    mf_top = g["MF勝ち確率"].idxmax()
    if mf_top != hon_i:
        g.at[mf_top, "妙味軸"] = "◎妙"
    return g


def main():
    df, pay = load()
    agg = collections.defaultdict(lambda: [0, 0, 0])   # key -> [投資,払戻,的中]
    band_agg = collections.defaultdict(lambda: [0, 0, 0])
    kind_agg = collections.defaultdict(lambda: [0, 0, 0])
    n_race = n_bet_race = 0
    pts_list = []
    verdict_cnt = collections.Counter()

    for rid, g in df.groupby("race_id"):
        if len(g) < 4:
            continue
        n_race += 1
        pdf = build_pdf(g)
        try:
            plan = kp._race_bet_plan(pdf)
            verdict_cnt[plan["判定"]] += 1
            rows = kp._build_bet_rows(pdf, rid)
        except Exception:
            continue
        if not rows:
            continue
        n_bet_race += 1
        pts_list.append(len(rows))
        for r in rows:
            amt = r["金額"]
            key = (rid, r["券種"], norm(r["券種"], r["組み合わせ"]))
            ret = pay.get(key, 0) / 100.0 * amt
            hit = 1 if ret > 0 else 0
            for a in (agg["all"], band_agg[r["判定"]], kind_agg[r["券種"]]):
                a[0] += amt; a[1] += ret; a[2] += hit

    def line(name, a):
        inv, ret, hit = a
        roi = ret / inv * 100 if inv else 0
        return f"  {name:12} 投資{int(inv):>9,}円 払戻{int(ret):>9,}円 収支{int(ret-inv):>+9,}円 回収{roi:6.1f}% 的中{hit}"

    print(f"=== 現行仕様 2025バックテスト（予算配分込み・JV実払戻）===")
    print(f"対象レース{n_race} / 購入レース{n_bet_race}（{n_bet_race/n_race*100:.0f}%）"
          f" / 平均{np.mean(pts_list):.0f}点·レース / 総購入点数{sum(pts_list):,}")
    print(f"判定分布: {dict(verdict_cnt)}")
    print("\n【全体】"); print(line("合計", agg["all"]))
    print("\n【判定帯別】")
    for b in ["勝負", "買い", "堅実", "少額"]:
        if band_agg[b][0]: print(line(b, band_agg[b]))
    print("\n【券種別】")
    for k in ["単勝", "複勝", "ワイド", "馬連", "馬単", "3連複", "3連単"]:
        if kind_agg[k][0]: print(line(k, kind_agg[k]))
    inv = agg["all"][0]
    print(f"\n日次目安: 購入{n_bet_race}R/年 → 1日≈{n_bet_race/(3144/36):.0f}R想定 "
          f"投資≈{int(inv/(n_bet_race or 1)):,}円/R")


if __name__ == "__main__":
    main()
