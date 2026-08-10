# -*- coding: utf-8 -*-
"""その日の予想と結果を1行1頭で履歴に積む。

なぜ必要か（2026-08-09）
  today_predictions.csv と today_results.csv は毎朝7時に消して作り直される。
  そのため「◎が何着だったか」「評価Aの馬の複勝率」「上位5印で馬券内を何頭
  拾えたか」といった分析は、その日のうちにしかできない。1日36レースでは
  何も判断できないのに、翌日にはデータが消える状態だった。

  accuracy_log.csv は1日1行の集計、prediction_record_v2.csv は◎○穴の3頭だけ。
  印ごとの成績や評価グレードの精度を後から検証するには足りない。

  ここで全頭ぶんを追記し、数ヶ月後にまとめて分析できるようにする。

出力: history_marks.csv（1行1頭・追記式）
  同じ日・同じレース・同じ馬番は上書きするので、何度実行しても二重にならない。

実行: 21:10（21:00の結果照合が終わったあと）に auto_predict_publish から呼ぶ。
      手動なら python archive_daily.py [YYYY-MM-DD]
"""
import os
import sys
from datetime import datetime

import pandas as pd

BASE_DIR = os.environ.get("KEIBA_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE_DIR, "history_marks.csv")

# 後の分析に要る列だけを残す。全列だと1日500行×300列で肥大するため。
KEEP = [
    "race_id", "馬番", "馬名", "jyo", "race_no", "距離", "馬場", "馬場状態", "クラス",
    "推奨ランク", "妙味", "妙味軸", "乖離", "買い指数", "購入推奨",
    "人気", "単勝オッズ",
    "勝ち確率", "連対確率", "複勝確率", "3着内確率", "単勝期待値",
    "MF予測順位", "MF勝ち確率", "MF複勝率", "MF複勝順位",
    "予測順位", "連対順位", "複勝順位",
    "能力_勝負", "能力_安定", "能力_末脚", "能力_先行", "能力_距離", "能力_実績",
]


def _is_today(path):
    """そのファイルが今日書かれたものか。"""
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).date() == \
            datetime.now().date()
    except Exception:
        return False


def build(date_str=None):
    """今日の予想と結果を突き合わせた1行1頭のDataFrameを返す。"""
    p = os.path.join(BASE_DIR, "today_predictions.csv")
    r = os.path.join(BASE_DIR, "today_results.csv")
    if not (os.path.exists(p) and os.path.exists(r)):
        print("  予想または結果のファイルが無い→スキップ")
        return None

    # 非開催日には積まない（2026-08-10追加）。
    #   このジョブは毎日21:10に無条件で動く。today_*.csv は開催日にしか
    #   書き換わらないので、ガードが無いと前回の開催日のデータを
    #   「今日の分」として毎晩積み直してしまう。半年で同じレースが数十回
    #   重複し、蓄積データが分析に使えなくなる。
    #   （同じ性質の事故が analyze_accuracy で実際に起きている: 8/2の結果が
    #     8/3・8/4にも記録された。日付ガードで直した経緯がある）
    #   date_str を明示したとき（過去分の再作成）はこの判定を通さない。
    if date_str is None and not (_is_today(p) and _is_today(r)):
        print("  当日のデータではない（非開催日）→スキップ")
        return None

    pred = pd.read_csv(p, dtype={"race_id": str})
    res = pd.read_csv(r, dtype={"race_id": str})
    if pred.empty or res.empty:
        print("  中身が空→スキップ")
        return None

    for df in (pred, res):
        df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce").astype("Int64")
    res["着順"] = pd.to_numeric(res["着順"], errors="coerce")

    cols = [c for c in KEEP if c in pred.columns]
    rcols = ["race_id", "馬番", "着順", "単勝", "複勝"] + \
        [c for c in ("確定単勝オッズ", "確定人気") if c in res.columns]
    d = pred[cols].merge(res[rcols], on=["race_id", "馬番"], how="inner")

    # 予想時のオッズと確定オッズのズレを残す（2026-08-10）。
    #   予想は7分前のオッズで買う馬を決めるが、検証(bet_cache)は確定オッズで
    #   判定している。8/9の1着馬36頭では中央値−8.8%動いており、期待値も
    #   同じだけずれる。どちらが正しいかを後から測れるようにしておく。
    #   「直前に買われる馬」を特徴量にする道も、この記録が前提になる。
    if "確定単勝オッズ" in d.columns:
        rec = pd.to_numeric(d.get("単勝オッズ"), errors="coerce")
        fin = pd.to_numeric(d["確定単勝オッズ"], errors="coerce")
        d["オッズ変化率"] = ((fin - rec) / rec * 100).round(1)
    if d.empty:
        print("  突き合わせ0件→スキップ")
        return None

    # 日付は予想日時からではなく実行日から取る（予想日時は更新のたび変わる）
    d.insert(0, "日付", date_str or datetime.now().strftime("%Y-%m-%d"))

    # どのモデルが出した予想かを残す（2026-08-09）。
    #   モデルは毎週火曜に再学習されるので、半年貯めたデータは「少しずつ違う
    #   モデルの出力」の寄せ集めになる。後から「この期間はモデルが変わった
    #   直後だった」と切り分けられるよう、学習物の更新日を列に入れておく。
    #   これが無いと、印の成績が変わった原因がモデル更新なのか偶然なのか
    #   永久に分からない。
    for col, fn in (("model版", "model.pkl"), ("MF版", "model_mf.pkl"),
                    ("較正版", "mf_calibrator.pkl")):
        fp = os.path.join(BASE_DIR, fn)
        d[col] = (datetime.fromtimestamp(os.path.getmtime(fp)).strftime("%Y-%m-%d")
                  if os.path.exists(fp) else "")
    d["着順"] = pd.to_numeric(d["着順"], errors="coerce")
    d["1着"] = (d["着順"] == 1).astype(int)
    d["2着内"] = (d["着順"] <= 2).astype(int)
    d["3着内"] = (d["着順"] <= 3).astype(int)
    return d


def archive(date_str=None):
    d = build(date_str)
    if d is None:
        return 0
    if os.path.exists(OUT):
        try:
            old = pd.read_csv(OUT, dtype={"race_id": str})
            old["馬番"] = pd.to_numeric(old["馬番"], errors="coerce").astype("Int64")
            d = pd.concat([old, d], ignore_index=True)
        except Exception as e:
            print(f"  既存の履歴を読めません（新規作成扱い）: {e}")
    # 同じ日・レース・馬番は最後を採用。再実行しても二重にならない。
    d = d.drop_duplicates(["日付", "race_id", "馬番"], keep="last")
    d.to_csv(OUT, index=False, encoding="utf-8-sig")
    days = d["日付"].nunique()
    races = d.groupby(["日付", "race_id"]).ngroups
    print(f"  履歴を更新: {len(d)}行 / {races}レース / {days}日分 → {OUT}")
    return races


if __name__ == "__main__":
    archive(sys.argv[1] if len(sys.argv) > 1 else None)
