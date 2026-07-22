# -*- coding: utf-8 -*-
"""オッズ変動特徴。keiba_predict.record_odds_snapshot が odds_history.csv に貯めた
時系列オッズ（朝／40分前／直前の各予想実行ぶん）から、レース×馬ごとの
「金の流れ」特徴を作る。モデルが一度も見ていない動的シグナル。

特徴（per race_id × 馬番）:
  朝単勝オッズ  = 最初のスナップの単勝オッズ
  直前単勝オッズ = 最後のスナップの単勝オッズ
  単勝変動率    = 直前 / 朝     （<1=支持集まり下落＝妙味/危険信号の両面）
  単勝下落幅    = 朝 - 直前
  人気変化      = 朝人気 - 直前人気（+なら人気上昇）
  複勝オッズ変動率 = 直前複勝 / 朝複勝（列があれば）
  スナップ数    = そのレースで記録できた時点数（信頼度）

使い方:
  import odds_movement_features as omf
  feat = omf.build_odds_movement_features()          # 全history から特徴表
  omf.status()                                        # 蓄積状況の確認
  <本番>: 予測時は race_features に (race_id,馬番) で左結合して使う（学習は蓄積後）
"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
HIST = os.path.join(BASE, "odds_history.csv")


def _load():
    if not os.path.exists(HIST):
        return pd.DataFrame()
    df = pd.read_csv(HIST, dtype={"race_id": str})
    df["記録時刻"] = pd.to_datetime(df["記録時刻"], errors="coerce")
    for c in ("単勝オッズ", "人気", "複勝オッズ_min", "馬番"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["記録時刻", "馬番"])


def build_odds_movement_features(hist=None):
    """odds_history から (race_id, 馬番) 単位のオッズ変動特徴を返す。"""
    df = hist if hist is not None else _load()
    if df.empty:
        return pd.DataFrame(columns=["race_id", "馬番"])
    df = df.sort_values("記録時刻")
    g = df.groupby(["race_id", "馬番"])
    first = g.first()
    last = g.last()
    out = pd.DataFrame(index=first.index)
    out["朝単勝オッズ"] = first["単勝オッズ"]
    out["直前単勝オッズ"] = last["単勝オッズ"]
    out["単勝変動率"] = last["単勝オッズ"] / first["単勝オッズ"]
    out["単勝下落幅"] = first["単勝オッズ"] - last["単勝オッズ"]
    if "人気" in df.columns:
        out["人気変化"] = first["人気"] - last["人気"]
    if "複勝オッズ_min" in df.columns:
        out["複勝変動率"] = last["複勝オッズ_min"] / first["複勝オッズ_min"]
    out["スナップ数"] = g.size()
    return out.reset_index()


def status():
    df = _load()
    if df.empty:
        print("odds_history.csv はまだ空です（レース日の予想実行で蓄積されます）。")
        return
    n_race = df["race_id"].nunique()
    snaps = df.groupby("race_id")["記録時刻"].nunique()
    multi = (snaps >= 2).sum()
    print(f"蓄積状況: {len(df):,}行 / {n_race}レース / 記録日数 "
          f"{df['記録時刻'].dt.date.nunique()}日")
    print(f"  2スナップ以上(変動が取れる)レース: {multi}/{n_race}")
    print(f"  平均スナップ数/レース: {snaps.mean():.1f}")
    print(f"  最新記録: {df['記録時刻'].max()}")


if __name__ == "__main__":
    status()
