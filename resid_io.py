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
    """軸の1頭を返す。gap が最大で、かつ gap >= gap_min のときだけ。

    しきい値の優先順位は 引数 > このファイルの AX_GAP。
    pkl の中の gap_min は見ない。pkl はモデルを作った時点の値を持っており、
    しきい値だけ変えたときに古い値が勝ってしまうため（2026-08-17に発生。
    AX_GAP を1.5にしたのに pkl の2.0が使われ、1.5〜2.0の馬が記録されなかった）。

    返り値: 1行のDataFrame、または None
    """
    if d is None or d.empty or "gap" not in d.columns:
        return None
    th = gap_min if gap_min is not None else AX_GAP
    g = pd.to_numeric(d["gap"], errors="coerce")
    if not g.notna().any():
        return None
    best = d.loc[[g.idxmax()]]
    if float(g.max()) < th:
        return None
    return best


# ── 確定した買い方（2026-08-17）─────────────────────────────────────
#   軸  : 残差モデルの gap が最大の1頭。gap >= AX_GAP
#         → 単勝 1点
#   ダートなら、さらに
#   相手: 軸以外で自分の gap >= MATE_GAP の馬。gap の大きい順に最大 MATE_MAX 頭
#         → ワイド（軸-相手）を追加
#   芝は単勝のみ。
#
#   なぜダートだけか
#     モデルの寄与（Benter基準比）が ダート6.3% に対し 芝1.3% で5倍違う。
#     これは回収率とは独立に model_diag.py で測った値なので、後付けではない。
#     実際、追加したワイドだけの成績は ダート174.9%（1,035点）。
#
#   検証値（walk-forward 5年・resid_gate.py）
#     2,926点  的中236  ROI 163.3%  95%区間[112.5, 226.2]  100%超 4/5年
#     年別 210 / 106 / 154 / 337 / 69%
#     順列検定 p=0.0000（判定5つ＋組み合わせを込みで150回、偽物の最大125.0%）
#
#   ⚠ 採用しなかったもの
#     軸gap>=3.0 で追加すると 180.5% と出るが、追加したワイドの的中が7本しかなく
#     直近3年は的中ゼロ。2021-2022の大穴3本で作られた数字なので採らない。
#     12頭以下で追加は 82.3% で逆効果。
#   2026-08-17 追記: 軸のしきい値を緩めた
#     厳しくすると数字は上がるが、少数の大穴に頼る形になり区間が広がる。
#     未見の2026年で上位3本を除いたときの成績が、それをはっきり示した。
#       軸gap>=2.0  2026年 268.1% → 上位3本除くと 31.4%
#       軸gap>=1.7  2026年 153.4% → 上位3本除くと 39.5%
#       軸gap>=1.5  2026年 115.6% → 上位3本除くと 57.2%
#     過去5年と2026年の一致度も、緩いほうが良い。
#       >=2.0  163.3% vs 268.1%（乖離105pt）
#       >=1.5  120.6% vs 115.6%（乖離  5pt）
#     年ごとの振れ幅も >=2.0 が268pt、>=1.5 が84pt。
#
#     そこで前向き検証では **1.5 と 1.7 の両方**を記録し、実測で決める。
#     記録は「軸gap>=1.5」で行い、1.7の成績はあとから絞り込んで出せる。
AX_GAP = 1.5             # 記録・買い判断のしきい値（緩めた側）。pklより優先
AX_GAP_TIGHT = 1.7       # もう一方の候補。paper_report が両方を集計する
MATE_GAP = 1.3
MATE_MAX = 3


def _is_dirt(d, pdf=None):
    """ダートかどうか。列でも attrs でも判定できるようにする。"""
    for col in ("is_turf",):
        if d is not None and col in d.columns:
            v = pd.to_numeric(d[col], errors="coerce").dropna()
            if len(v):
                return bool(v.iloc[0] == 0)
    for src in (d, pdf):
        if src is None:
            continue
        t = getattr(src, "attrs", {}).get("turf")
        if t is not None:
            return str(t).startswith(("ダ", "ダート"))
        if "馬場" in getattr(src, "columns", []):
            v = src["馬場"].dropna()
            if len(v):
                return str(v.iloc[0]).startswith(("ダ", "ダート"))
    return None                      # 判定できない → ワイドは足さない


def pick_bets(d, model=None, pdf=None):
    """このレースで買う買い目を全部返す。

    返り値: [{"券種","組み合わせ","馬番","馬名","単勝オッズ","gap","役割"}, ...]
            買わないときは空リスト。
    ⚠ 組み合わせの馬番は2桁ゼロ埋め。払戻表が "09-14" 形式なので、
      揃えないと1桁馬番の買い目が照合できない（2026-08-16の事故）。
    """
    ax = pick_bet(d, model=model)
    if ax is None:
        return []
    a = ax.iloc[0]
    an = _bn2(a.get("馬番"))
    if an is None:
        return []
    out = [{"券種": "単勝", "組み合わせ": an, "馬番": a.get("馬番"),
            "馬名": a.get("馬名"), "単勝オッズ": a.get("単勝オッズ"),
            "gap": float(a["gap"]), "役割": "軸"}]
    if _is_dirt(d, pdf) is not True:
        return out                    # 芝・判定不能は単勝のみ
    g = pd.to_numeric(d["gap"], errors="coerce")
    rest = d.drop(index=ax.index)
    rest = rest[pd.to_numeric(rest["gap"], errors="coerce") >= MATE_GAP]
    if rest.empty:
        return out
    for _, m in rest.nlargest(min(MATE_MAX, len(rest)), "gap").iterrows():
        bn = _bn2(m.get("馬番"))
        if bn is None or bn == an:
            continue
        out.append({"券種": "ワイド", "組み合わせ": f"{min(an,bn)}-{max(an,bn)}",
                    "馬番": m.get("馬番"), "馬名": m.get("馬名"),
                    "単勝オッズ": m.get("単勝オッズ"),
                    "gap": float(m["gap"]), "役割": "相手"})
    return out


def _bn2(v):
    """馬番を2桁ゼロ埋めの文字列にする。数値化できなければ None。"""
    n = pd.to_numeric(v, errors="coerce")
    if pd.isna(n):
        return None
    return str(int(n)).zfill(2)
