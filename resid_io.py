# -*- coding: utf-8 -*-
"""残差モデルの読み込みと予測（本番・検証の共通窓口）（2026-08-17）

なぜ窓口を1つにするか
  2026-08-16〜17に「検証で見た数字」と「本番が実際に買うもの」がズレる事故を
  5回起こした（馬番ゼロ埋め / 軸オッズ条件 / 距離分割 / 順位の較正前後 /
  検証と本番でモデルが違う）。原因はどれも、同じ計算を2か所に書いたこと。

  残差モデルでは最初から窓口を1つにする。gap の作り方も買い判断も
  ここにしか書かない。train_resid.py も keiba_predict.py もここを呼ぶ。

使い方:
    from resid_io import load_model, predict_gap, pick_bet
    m = load_model()
    df = predict_gap(m, pdf)        # gap / p / EV 列が付く
    bet = pick_bet(df)              # 買う1頭（無ければ None）
"""
import os
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-9
MODEL_PATH = os.path.join(BASE_DIR, "model_resid.pkl")

_M = None
_LOADED = False


def load_model(path=MODEL_PATH):
    """1度だけ読む。無ければ None（呼び出し側は予測なしとして扱うこと）。"""
    global _M, _LOADED
    if not _LOADED:
        _LOADED = True
        try:
            with open(path, "rb") as fh:
                _M = pickle.load(fh)
            print(f"  残差モデルを読込（{_M['years'][0]}〜{_M['years'][1]} / "
                  f"{_M['n']:,}頭 / シード{_M['seeds']}）")
        except FileNotFoundError:
            print("  残差モデルなし（train_resid.py で作成できます）")
        except Exception as e:
            print(f"  残差モデルの読込に失敗: {e}")
    return _M


def market_prob(odds, race_key):
    """オッズからレース内の市場確率。控除率は正規化で消える。"""
    o = pd.to_numeric(pd.Series(odds), errors="coerce")
    inv = 1.0 / o.clip(lower=1.01)
    return inv / inv.groupby(pd.Series(race_key, index=inv.index)).transform("sum")


def predict_gap(model, df, race_col="race_id", odds_col="単勝オッズ"):
    """gap（モデル予測確率 ÷ 市場確率）などを付けて返す。

    gap の意味: 1.0 なら市場と同じ評価、2.0 なら市場の2倍強いと見ている。
    レース内の gap の順位はオッズに依存しない（EV ∝ exp(f) のため）。
    ただし gap の絶対値はレース全体のオッズ分布で決まるので、
    しきい値判定はオッズが確定に近いほど安定する。
    """
    if model is None or df is None or df.empty:
        return None
    cols = model["use_cols"]
    miss = [c for c in cols if c not in df.columns]
    if miss:
        print(f"  残差モデル: 特徴量が{len(miss)}列足りません（例 {miss[:3]}）")
        return None
    o = pd.to_numeric(df[odds_col], errors="coerce")
    if o.notna().sum() < 2 or (o.fillna(0) <= 0).all():
        return None                       # オッズ未取得。予測できない
    d = df.copy()
    key = d[race_col] if race_col in d.columns else pd.Series("x", index=d.index)
    d["_q"] = market_prob(o, key).values
    d["_lq"] = np.log(d["_q"].clip(EPS))
    X = d[cols].apply(pd.to_numeric, errors="coerce")
    f = np.mean([m.predict(X, raw_score=True) for m in model["models"]], axis=0)
    d["_f"] = f
    sc = d["_f"] + d["_lq"]
    e = np.exp(sc - sc.groupby(key).transform("max"))
    d["残差確率"] = (e / e.groupby(key).transform("sum")).values
    d["gap"] = d["残差確率"] / d["_q"]
    d["残差EV"] = d["残差確率"] * o
    return d


def pick_bet(d, gap_min=None, model=None):
    """買う1頭を返す。事前登録した買い方そのもの。

      各レースで gap が最大の1頭。ただし gap >= gap_min のときだけ。
      他の条件は付けない（付けると検証と別の買い方になる）。

    返り値: 1行のDataFrame、または None
    """
    if d is None or d.empty or "gap" not in d.columns:
        return None
    th = gap_min if gap_min is not None else (model or {}).get("gap_min", 2.0)
    g = pd.to_numeric(d["gap"], errors="coerce")
    if not g.notna().any():
        return None
    best = d.loc[[g.idxmax()]]
    if float(g.max()) < th:
        return None
    return best
