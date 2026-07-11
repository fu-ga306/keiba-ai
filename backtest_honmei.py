# -*- coding: utf-8 -*-
"""
backtest_honmei.py
──────────────────
◎（本命軸）選定ロジックのバックテスト基盤。

keiba_predict.py の総合スコア＝◎選定ロジックを 2025年テストデータ全体で再現し、
「blend重み(MF比率) / EVペナルティ / 人気bonus」などを振って
◎の 勝率・複勝率・単勝回収率・複勝回収率 を客観比較する。

入力:
  model_result.csv     … 通常winモデル(race_id,馬名,着順_num,単勝オッズ,人気,勝ち確率,単勝期待値)
  model_mf_result.csv  … MFモデル(race_id,馬名,MF勝率,MF複勝率 ...)
    ※どちらも 2025年テストセット（学習に未使用）

◎の定義:
  keiba_predict.py と同じ。総合スコア最上位1頭を◎とする。
    blend      = w_mf*MF勝率(正規化) + (1-w_mf)*通常勝率(正規化)
    ev         = MF勝率 * 単勝オッズ - 1      （keiba_predict.py:898 と同じ）
    ev_adj     = ev>=ev_th のとき減点、それ以外1.0（EVペナルティ）
    ninki_bonus= MF上位3頭 かつ 2-4番人気 に +bonus
    総合スコア  ∝ (blend+ninki_bonus) * ev_adj
  ※該当戦略(20点bonus)は 距離/前走間隔 が本CSVに無いため 戦略A(予測順位1&1.5<=odds<=20)
    のみ近似。詳細は keiba_predict.py:_check_strategy 参照。

使い方:
  python backtest_honmei.py            # 主要プリセットを比較
  python backtest_honmei.py sweep      # MF比率を 0.0〜1.0 でスイープ
"""
import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WIN_CSV  = os.path.join(BASE_DIR, "model_result.csv")
MF_CSV   = os.path.join(BASE_DIR, "model_mf_result.csv")


def load_merged():
    """通常モデル結果とMFモデル結果を race_id+馬名 で突合して返す。"""
    win = pd.read_csv(WIN_CSV)
    mf  = pd.read_csv(MF_CSV)
    for d in (win, mf):
        d["race_id"] = d["race_id"].astype(str)
        d["馬名"]    = d["馬名"].astype(str).str.strip()

    win = win.rename(columns={"勝ち確率": "通常勝率"})
    keep_win = ["race_id", "馬名", "着順_num", "単勝オッズ", "人気", "通常勝率", "予測順位"]
    keep_win = [c for c in keep_win if c in win.columns]
    keep_mf  = ["race_id", "馬名", "MF勝率", "MF複勝率"]
    keep_mf  = [c for c in keep_mf if c in mf.columns]

    df = win[keep_win].merge(mf[keep_mf], on=["race_id", "馬名"], how="inner")
    df = df.dropna(subset=["着順_num", "単勝オッズ", "通常勝率", "MF勝率"])
    # 明らかな異常オッズ除去（取消・0倍など）
    df = df[df["単勝オッズ"] >= 1.0]
    return df


def _norm(s):
    tot = s.sum()
    return s / tot if tot > 0 else pd.Series(np.ones(len(s)) / len(s), index=s.index)


def select_honmei(df, *, w_mf=0.6, ev_penalty=True, ev_th=0.1,
                  ninki_bonus=0.05, strat_bonus=True, axis_mode="score"):
    """レースごとに◎（軸1頭）を選び、選ばれた行だけを返す。

    axis_mode:
      "score"  … 総合スコア最上位（keiba_predict.py準拠）
      "ev"     … blend上位5頭のうち単勝EV最大（回収率重視の◎）
      "mf"     … MF勝率最上位のみ
      "normal" … 通常勝率最上位のみ
    """
    picks = []
    for rid, g in df.groupby("race_id"):
        g = g.copy()
        n = len(g)
        mf_norm  = _norm(g["MF勝率"])
        win_norm = _norm(g["通常勝率"])
        blend    = w_mf * mf_norm + (1 - w_mf) * win_norm

        ev = g["MF勝率"] * g["単勝オッズ"] - 1
        if ev_penalty:
            ev_adj = np.where(ev >= ev_th,
                              np.maximum(0.55, 1.0 - ev.clip(0, 0.5)), 1.0)
        else:
            ev_adj = np.ones(n)

        pop = pd.to_numeric(g["人気"], errors="coerce").fillna(8)
        mf_rank = g["MF勝率"].rank(ascending=False)
        nbonus = ((mf_rank <= 3) & (pop >= 2) & (pop <= 4)).astype(float) * ninki_bonus
        blend_adj = _norm(blend + nbonus)

        score = blend_adj * ev_adj
        ai_rank = pd.Series(score, index=g.index).rank(ascending=False)

        # 戦略A近似（予測順位1 & 1.5<=odds<=20）→ +20点相当
        strat = np.zeros(n)
        if strat_bonus and "予測順位" in g.columns:
            a = (g["予測順位"] == 1) & (g["単勝オッズ"] >= 1.5) & (g["単勝オッズ"] <= 20)
            strat = a.astype(float).values * 20
        total = (1 - (ai_rank - 1) / n) * 80 + strat

        if axis_mode == "mf":
            idx = g["MF勝率"].idxmax()
        elif axis_mode == "normal":
            idx = g["通常勝率"].idxmax()
        elif axis_mode == "ev":
            top5 = pd.Series(blend_adj, index=g.index).nlargest(min(5, n)).index
            idx = ev.loc[top5].idxmax()
        else:  # score
            idx = pd.Series(total, index=g.index).idxmax()
        picks.append(g.loc[[idx]])
    return pd.concat(picks, ignore_index=True)


def report(name, honmei):
    n = len(honmei)
    chaku = honmei["着順_num"]
    win  = (chaku == 1).mean() * 100
    ren  = (chaku <= 2).mean() * 100
    fuku = (chaku <= 3).mean() * 100
    # 単勝回収率: 的中時オッズ払い戻し / ベット総額(=n*1)
    tan_roi = honmei.loc[chaku == 1, "単勝オッズ"].sum() / n * 100
    # 複勝回収率(推定): 複勝オッズ≒単勝/4 で近似（実オッズは別途payout_data参照）
    fuku_hit = honmei[chaku <= 3]
    fuku_roi = (fuku_hit["単勝オッズ"] / 4).clip(lower=1.05).sum() / n * 100
    avg_pop = pd.to_numeric(honmei["人気"], errors="coerce").mean()
    fav_share = (pd.to_numeric(honmei["人気"], errors="coerce") == 1).mean() * 100
    print(f"  {name:<26} n={n:4d}  勝率{win:5.1f}%  複勝{fuku:5.1f}%  "
          f"単回収{tan_roi:6.1f}%  複回収(推){fuku_roi:6.1f}%  "
          f"◎平均人気{avg_pop:4.1f}  1人気率{fav_share:4.0f}%")


def report_by_pop(name, honmei):
    """◎の人気帯別 勝率・単勝回収率（的中と回収の分離を見る）"""
    h = honmei.copy()
    h["人気"] = pd.to_numeric(h["人気"], errors="coerce")
    bands = [(1, 1, "1番人気"), (2, 3, "2-3番人気"),
             (4, 6, "4-6番人気"), (7, 99, "7番人気以下")]
    print(f"  [{name}] 人気帯別")
    for lo, hi, lbl in bands:
        g = h[(h["人気"] >= lo) & (h["人気"] <= hi)]
        if len(g) == 0:
            continue
        win = (g["着順_num"] == 1).mean() * 100
        roi = g.loc[g["着順_num"] == 1, "単勝オッズ"].sum() / len(g) * 100
        print(f"    {lbl:<10} {len(g):4d}回  勝率{win:5.1f}%  単回収{roi:6.1f}%")


def select_myumi(df, *, min_pop=1, exclude_honmei=True):
    """妙味◎（回収率用軸）候補を選ぶ。
    ルール: 各レースでMF勝率最上位。min_pop で人気下限を課す（1番人気を除く等）。
    exclude_honmei=True のとき、通常◎(総合スコア最上位)と同一馬なら「妙味なし」として除外。
    戻り値: 妙味◎に選ばれた行のみ（レースによっては0行）。
    """
    honmei = select_honmei(df, w_mf=0.6, ev_penalty=False, ninki_bonus=0.0)
    honmei_map = dict(zip(honmei["race_id"], honmei["馬名"]))
    picks = []
    for rid, g in df.groupby("race_id"):
        pop = pd.to_numeric(g["人気"], errors="coerce").fillna(99)
        cand = g[pop >= min_pop]
        if len(cand) == 0:
            continue
        idx = cand["MF勝率"].idxmax()
        row = g.loc[idx]
        if exclude_honmei and honmei_map.get(rid) == row["馬名"]:
            continue  # ◎と同じ馬 → 妙味印を出さない（軸に一本化）
        picks.append(g.loc[[idx]])
    return pd.concat(picks, ignore_index=True) if picks else g.iloc[0:0]


def main():
    df = load_merged()
    n_races = df["race_id"].nunique()
    print(f"バックテスト対象: {n_races}レース / {len(df)}頭（2025テストセット）\n")

    print("=" * 100)
    print("【妙味◎】回収率用軸ルールの検証（追加印・◎とは別馬）")
    print("=" * 100)
    report("妙味◎: MF最上位(◎重複除く)",       select_myumi(df, min_pop=1))
    report("妙味◎: MF最上位∩人気≥2",           select_myumi(df, min_pop=2))
    report("妙味◎: MF最上位∩人気≥3",           select_myumi(df, min_pop=3))
    print("  （参考: 現行◎ = 下表『現行』行。妙味◎は◎と別馬のみ集計＝実際に追加される印）\n")

    print("=" * 100)
    print("◎選定ロジック比較（総合スコア方式・パラメータ違い）")
    print("=" * 100)
    configs = [
        ("現行(MF60%+EVペナ+人気B)", dict(w_mf=0.6, ev_penalty=True,  ninki_bonus=0.05)),
        ("EVペナルティOFF",          dict(w_mf=0.6, ev_penalty=False, ninki_bonus=0.05)),
        ("人気bonusもOFF",           dict(w_mf=0.6, ev_penalty=False, ninki_bonus=0.0)),
        ("MF40%へ低減",              dict(w_mf=0.4, ev_penalty=False, ninki_bonus=0.0)),
        ("MF80%へ増加",              dict(w_mf=0.8, ev_penalty=False, ninki_bonus=0.0)),
    ]
    for name, kw in configs:
        report(name, select_honmei(df, **kw))

    print("\n" + "=" * 100)
    print("軸モード比較（単一モデル vs ブレンド vs 回収率重視）")
    print("=" * 100)
    report("通常モデル単独",   select_honmei(df, axis_mode="normal"))
    report("MFモデル単独",     select_honmei(df, axis_mode="mf"))
    report("ブレンド(EVペナOFF)", select_honmei(df, w_mf=0.6, ev_penalty=False, ninki_bonus=0.0))
    report("回収率重視(blend上位5→EV最大)", select_honmei(df, w_mf=0.6, axis_mode="ev"))

    print("\n" + "=" * 100)
    print("的中用◎ vs 回収率用◎ の人気帯別内訳")
    print("=" * 100)
    report_by_pop("的中用◎: ブレンドEVペナOFF",
                  select_honmei(df, w_mf=0.6, ev_penalty=False, ninki_bonus=0.0))
    report_by_pop("回収率用◎: blend上位5→EV最大",
                  select_honmei(df, w_mf=0.6, axis_mode="ev"))


def sweep():
    df = load_merged()
    print(f"MF比率スイープ（EVペナルティOFF, 人気bonus OFF）  対象{df['race_id'].nunique()}レース\n")
    for w in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        report(f"w_mf={w:.1f}", select_honmei(df, w_mf=w, ev_penalty=False, ninki_bonus=0.0))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        sweep()
    else:
        main()
