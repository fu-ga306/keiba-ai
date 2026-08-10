# -*- coding: utf-8 -*-
"""同じ競馬場の終わったレースの着順・払戻を集めてダッシュボードに出す。

要望(2026-08-07): 40分前の更新時に、その競馬場の前のレース結果を取得して表示。
各馬の着順と払戻を見たい。

⚠️ 他のスクレイピングに干渉しないことが設計の要（2026-08-07 仕様見直し）
  実測: 3場開催（36レース）の日は、40分前ジョブ[T-45,T-30]と7分前ジョブ[T-12,T]で
  10:00〜16:35が完全に埋まる。レース時間中に「空いている時刻」は1分もない。
  つまり結果取得のために独立したタイマーを昼間に回すと、必ずどれかと衝突する。

  そこで取得経路を2本だけにし、どちらも既存の通信の合間に相乗りさせる:

  A) 相乗り（昼間）… 40分前ジョブの最後に update_for_race() を呼ぶ。
     scheduleは単一スレッドで順番に実行するので、予想処理と同時には走らない。
     同じ競馬場のまだ取っていない終了レースを最大2件、2秒あけて逐次取得。
     ダッシュボードの「前レース結果」はこれで埋まる。

  B) 後片付け（レース終了後）… sweep() を17:00〜20:40の間に10分おき。
     Aは「後続レースがあるレース」しか拾えないので最終レース(12R)が漏れる。
     終了後は 16:35〜22:00 に325分の空きがあるのでここで安全に回収できる。
     さらに念のため、どのレースの発走±15分にも入っていないことを毎回確認する。
     21:00の結果照合より前に終わらせるので、照合もキャッシュを使えて速くなる。

  共通: 並列にしない。既に取った行は二度と取りに行かない。
        1日の取得回数は最大でもレース数と同じ。

保存: today_results.csv（race_id, 馬番, 馬名, 着順, 単勝, 複勝）
"""
import json
import os
import time
from datetime import datetime

import pandas as pd

BASE_DIR = os.environ.get("KEIBA_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE_DIR, "today_results.csv")
LOG = os.path.join(BASE_DIR, "today_results.log")
TIMES = os.path.join(BASE_DIR, "today_race_times.json")   # 発走時刻（衝突判定用）
SLEEP_SEC = 2.0          # 取得の間に必ず入れる待機
MAX_PER_RUN = 4          # 相乗り時に取りに行く上限（取りこぼしは次のジョブが拾う）
                         # 2026-08-09: 2だと遅延から復帰できずダッシュボードが
                         # 何レースも空欄のままになった。2秒あけて4件でも1ジョブ
                         # あたり8秒。昼休みやジョブ停止のあとに追いつけるよう上げる。
SWEEP_PER_RUN = 3        # 後片付け時の上限
QUIET_MARGIN = 15        # 発走の前後この分数は取得しない
SWEEP_FROM = (17, 0)     # 後片付けの開始
SWEEP_TO = (20, 40)      # 21:00の結果照合に被らないよう終える
CONFIRM_WAIT = 12        # 発走からこの分数は結果が確定しないので取りに行かない


def _log(msg):
    """画面と today_results.log の両方に出す。

    auto_predict_publish はタスクスケジューラからコンソールなしで起動されるため、
    print だけでは出力がどこにも残らない（2026-08-07に判明）。動いたかどうかを
    後から確かめられるよう、この機能のログはファイルにも書く。
    """
    line = f"[{datetime.now().strftime('%m/%d %H:%M:%S')}] {msg}"
    print(f"  {line}")
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _load():
    if not os.path.exists(OUT):
        return pd.DataFrame(columns=["race_id", "馬番", "馬名", "着順", "単勝", "複勝",
                                     "確定単勝オッズ", "確定人気"])
    try:
        return pd.read_csv(OUT, dtype={"race_id": str, "馬番": str})
    except Exception:
        return pd.DataFrame(columns=["race_id", "馬番", "馬名", "着順", "単勝", "複勝",
                                     "確定単勝オッズ", "確定人気"])


def _same_venue_before(race_id, bets_or_pred):
    """同じ競馬場・同じ開催日で、このレースより前のrace_idを返す。"""
    rid = str(race_id)
    head = rid[:10]                      # 年+場+回+日
    try:
        rno = int(rid[10:12])
    except ValueError:
        return []
    out = []
    for r in bets_or_pred["race_id"].astype(str).unique():
        if r[:10] != head:
            continue
        try:
            n = int(r[10:12])
        except ValueError:
            continue
        if n < rno:
            out.append(r)
    return sorted(out)


def fetch_one(rid):
    """1レース分の着順＋払戻を取る。取れなければNone。"""
    try:
        from result_tracker import get_race_result
    except Exception as e:
        _log(f"結果取得モジュールを読めません: {e}")
        return None
    res = get_race_result(rid)
    if res is None or len(res) == 0:
        return None
    df = res.copy()
    for c in ("馬番", "馬名", "着順_num"):
        if c not in df.columns:
            return None
    # 確定オッズ・確定人気も一緒に残す（2026-08-10）。
    #   予想時のオッズ（7分前）しか記録が無いと、検証と実運用のズレを測れない。
    #   結果ページに載っているので追加のアクセスは要らない。
    _extra = [c for c in ("確定単勝オッズ", "確定人気") if c in df.columns]
    df = df[["馬番", "馬名", "着順_num"] + _extra].rename(
        columns={"着順_num": "着順"})
    for c in ("確定単勝オッズ", "確定人気"):
        if c not in df.columns:
            df[c] = pd.NA
    df["race_id"] = rid
    df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce").astype("Int64").astype(str)

    # 払戻（単勝・複勝）を付ける。payout_sourceはJV優先なので追加の負荷は小さい。
    tan, fuku = {}, {}
    try:
        from payout_source import get_payout
        pay = get_payout(rid)
        if pay and pay != "BLOCKED":
            for p in pay:
                k = str(p.get("券種", ""))
                combo = str(p.get("組み合わせ", "")).lstrip("0") or "0"
                amt = p.get("払戻金")
                if k == "単勝":
                    tan[combo] = amt
                elif k == "複勝":
                    fuku[combo] = amt
    except Exception as e:
        _log(f"払戻取得スキップ({rid}): {str(e)[:60]}")
    df["単勝"] = df["馬番"].map(lambda b: tan.get(str(b).lstrip("0") or "0"))
    df["複勝"] = df["馬番"].map(lambda b: fuku.get(str(b).lstrip("0") or "0"))
    return df[["race_id", "馬番", "馬名", "着順", "単勝", "複勝",
               "確定単勝オッズ", "確定人気"]]


def update_for_race(race_id, pred_df=None):
    """race_id と同じ競馬場の、まだ取っていない前レースの結果を取りに行く。

    40分前ジョブの最後から呼ばれる相乗り取得(A)。1回で最大 MAX_PER_RUN 件、
    取りこぼしは次の40分前ジョブか、レース後の sweep() が拾う。
    """
    if pred_df is None:
        p = os.path.join(BASE_DIR, "today_predictions.csv")
        if not os.path.exists(p):
            return
        pred_df = pd.read_csv(p, usecols=["race_id"], dtype={"race_id": str})
    have = _load()
    done = set(have["race_id"].astype(str).unique())
    todo = [r for r in _same_venue_before(race_id, pred_df) if r not in done]
    if not todo:
        return
    # 直近のレースから順に取る（見たいのは直前の結果なので）
    todo = sorted(todo, reverse=True)[:MAX_PER_RUN]
    got = []
    for i, rid in enumerate(todo):
        if i:
            time.sleep(SLEEP_SEC)     # 逐次。並列にはしない
        try:
            d = fetch_one(rid)
        except Exception as e:
            _log(f"結果取得エラー({rid}): {str(e)[:60]}")
            continue
        if d is None or d.empty:
            continue                   # まだ確定していない等
        got.append(d)
        _log(f"結果取得: {rid} ({len(d)}頭)")
    if not got:
        return
    # 空のDataFrameを混ぜるとpandasが列の型を決められず警告を出すので除く
    parts = ([have] if not have.empty else []) + got
    new = pd.concat(parts, ignore_index=True)
    new = new.drop_duplicates(["race_id", "馬番"], keep="last")
    new.to_csv(OUT, index=False, encoding="utf-8-sig")
    _log(f"相乗り保存: {new['race_id'].nunique()}レース分")


# ── B) 後片付け（レース終了後に残りを回収） ──────────────────────────────
def _race_times():
    """{race_id: 発走の分(0-1439)}。setup_schedule が書き出す。

    今日書かれたものでなければ空を返す（2026-08-10追加）。
    このファイルは開催日にしか更新されないので、非開催日には前回の開催日の
    時刻が残る。鮮度を見ないと sweep が「前回のレースがまだ取れていない」と
    判断して取りに行きうる。取得済みなら実害は出ないが、today_results.csv が
    何かの拍子に消えていると、非開催日に過去レースを取りに行くことになる。
    """
    if not os.path.exists(TIMES):
        return {}
    try:
        if datetime.fromtimestamp(os.path.getmtime(TIMES)).date() != \
                datetime.now().date():
            return {}
    except Exception:
        return {}
    try:
        with open(TIMES, encoding="utf-8") as f:
            return {str(k): int(v) for k, v in json.load(f).items()}
    except Exception:
        return {}


def save_race_times(race_info):
    """{race_id: '15:45'} を分に直して保存する。setup_schedule から呼ぶ。"""
    import re
    out = {}
    for rid, t in race_info.items():
        m = re.search(r"(\d{1,2}):(\d{2})", str(t))
        if m:
            out[str(rid)] = int(m.group(1)) * 60 + int(m.group(2))
    with open(TIMES, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    if out:
        last = max(out.values())
        _log(f"発走時刻を保存: {len(out)}レース（最終 {last // 60}:{last % 60:02d}）")
        # 前日までの結果が残っているとダッシュボードに古い行が混ざるので、
        # 当日のレース以外は落とす（毎朝7:05の登録時に1回だけ通る）。
        have = _load()
        if not have.empty:
            keep = have[have["race_id"].astype(str).isin(out.keys())]
            if len(keep) != len(have):
                keep.to_csv(OUT, index=False, encoding="utf-8-sig")
                _log(f"前日分を整理: {len(have)}行 → {len(keep)}行")
    return out


def is_quiet(now=None, times=None):
    """今、他のスクレイピングが走っていないと言える時刻か。

    どのレースの発走±QUIET_MARGIN分にも入っていなければ静か、と判定する。
    40分前ジョブ(T-40)と7分前ジョブ(T-7)は、この窓の外に出ることはない。
    """
    now = now or datetime.now()
    cur = now.hour * 60 + now.minute
    if not (SWEEP_FROM[0] * 60 + SWEEP_FROM[1] <= cur <= SWEEP_TO[0] * 60 + SWEEP_TO[1]):
        return False
    times = _race_times() if times is None else times
    for t in times.values():
        # 40分前ジョブも7分前ジョブも [T-45, T+margin] の中で動く
        if t - 45 - QUIET_MARGIN <= cur <= t + QUIET_MARGIN:
            return False
    return True


def sweep():
    """レース終了後に、まだ取れていない当日レースの結果を回収する。

    相乗り(A)では後続レースのない最終レースが漏れるので、その受け皿。
    """
    if not is_quiet():
        return 0
    times = _race_times()
    if not times:
        return 0
    now = datetime.now()
    cur = now.hour * 60 + now.minute
    have = _load()
    done = set(have["race_id"].astype(str).unique())
    todo = sorted(
        [r for r, t in times.items() if r not in done and cur >= t + CONFIRM_WAIT],
        reverse=True,
    )[:SWEEP_PER_RUN]
    if not todo:
        return 0
    _log(f"後片付け開始: 残り{len(todo)}件")
    got = []
    for i, rid in enumerate(todo):
        if i:
            time.sleep(SLEEP_SEC)
        try:
            d = fetch_one(rid)
        except Exception as e:
            _log(f"結果取得エラー({rid}): {str(e)[:60]}")
            continue
        if d is not None and not d.empty:
            got.append(d)
            _log(f"結果取得: {rid} ({len(d)}頭)")
    if not got:
        return 0
    parts = ([have] if not have.empty else []) + got
    new = pd.concat(parts, ignore_index=True)
    new = new.drop_duplicates(["race_id", "馬番"], keep="last")
    new.to_csv(OUT, index=False, encoding="utf-8-sig")
    _log(f"後片付け保存: {new['race_id'].nunique()}レース分")
    return len(got)


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        print(f"静かな時刻か: {is_quiet()}")
        print(f"取得: {sweep()}件")
    elif len(sys.argv) > 1:
        update_for_race(sys.argv[1])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
