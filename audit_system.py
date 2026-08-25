# -*- coding: utf-8 -*-
"""システム全体の欠損・漏れを機械的に洗い出す（2026-08-18）

なぜ作るか
  週次更新の Step0 が --skip-horse で血統取得を飛ばしており、コメントには
  「別タスクで実施」とあったのにその別タスクが存在せず、horse_master.csv が
  1か月半止まっていた。こういう「誰も見ていない穴」を人手で探すのは無理。

  同じ形の穴が他にもないかを、毎回同じ手順で調べられるようにする。

見るもの
  ① データの鮮度   … 各ファイルがいつ更新されたか。止まっているものはないか
  ② 依存関係の整合 … 出力が入力より古くないか（作り直しが必要な状態でないか）
  ③ 設定の二重定義 … 同じしきい値が複数の場所に書かれていないか
  ④ 飛ばされた工程 … --skip 系のフラグが立っていないか
  ⑤ 参照切れ      … コードが読むファイルが存在するか
  ⑥ 週次の網羅性  … 主要スクリプトが週次工程に入っているか

実行: python audit_system.py
"""
import sys

for _s in (sys.stdout, sys.stderr):   # cp932環境でのUnicodeEncodeError→異常終了を防ぐ
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import re
import subprocess
import warnings
from datetime import datetime, timedelta

import pandas as pd

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
NG = []


def log(m):
    print(m, flush=True)


def ng(sev, msg):
    NG.append((sev, msg))


def age(p):
    f = os.path.join(BASE, p)
    if not os.path.exists(f):
        return None
    return (datetime.now() - datetime.fromtimestamp(os.path.getmtime(f))).days


def main():
    log("=" * 70)
    log("  ① データの鮮度")
    log("=" * 70)
    # (ファイル, 期待する更新間隔[日], 説明)
    FRESH = [
        ("race_data_clean.csv", 10, "前処理済みレースデータ"),
        ("race_features.csv", 10, "特徴量"),
        ("horse_master.csv", 10, "血統マスタ"),
        ("sire_stats_father.csv", 10, "種牡馬成績(本番用)"),
        ("sire_stats_father_train.csv", 40, "種牡馬成績(学習用)"),
        ("jv_payouts.csv", 40, "JRA-VAN払戻"),
        ("model_resid.pkl", 10, "残差モデル"),
        ("model_mf_parts", 40, "MFモデル"),
        ("grade_calib.pkl", 40, "評価の較正器"),
        ("today_predictions.csv", 3, "当日予想"),
        ("odds_history.csv", 3, "オッズ履歴"),
    ]
    log(f"  {'ファイル':<30}{'経過':>7}{'目安':>7}  判定")
    for p, lim, desc in FRESH:
        a = age(p)
        if a is None:
            log(f"  {p:<30}{'---':>7}{lim:>7}  ⚠ 存在しない（{desc}）")
            ng("高", f"{p} が存在しない（{desc}）")
            continue
        ok = a <= lim
        log(f"  {p:<30}{a:>6}日{lim:>6}日  {'○' if ok else '⚠ 古い'}（{desc}）")
        if not ok:
            ng("中" if a < lim * 3 else "高", f"{p} が{a}日更新されていない（{desc}）")

    log("\n" + "=" * 70)
    log("  ② 依存関係（出力が入力より古くないか）")
    log("=" * 70)
    DEPS = [
        ("race_data_clean.csv", "race_features.csv", "元データ→特徴量"),
        ("horse_master.csv", "sire_stats_father.csv", "血統マスタ→種牡馬成績"),
        ("sire_stats_father.csv", "race_features.csv", "種牡馬成績→特徴量"),
        ("race_features.csv", "model_resid.pkl", "特徴量→残差モデル"),
        ("race_features.csv", "bet_cache_2025.csv", "特徴量→検証キャッシュ"),
    ]
    log(f"  {'入力':<28}{'出力':<28}判定")
    for src, dst, desc in DEPS:
        sp, dp = os.path.join(BASE, src), os.path.join(BASE, dst)
        if not (os.path.exists(sp) and os.path.exists(dp)):
            log(f"  {src:<28}{dst:<28}― どちらか無い")
            continue
        st, dt = os.path.getmtime(sp), os.path.getmtime(dp)
        ok = dt >= st
        d = (st - dt) / 86400
        log(f"  {src:<28}{dst:<28}{'○' if ok else f'⚠ 出力が{d:.1f}日古い'}（{desc}）")
        if not ok:
            ng("中", f"{dst} が {src} より{d:.1f}日古い（{desc}）。作り直しが要る")

    log("\n" + "=" * 70)
    log("  ②b 対になるファイルの片方だけ古くないか")
    log("=" * 70)
    log("  本番用と学習用など、同時に作られるべきファイルの組を見る。")
    log("  片方だけ更新されるのは「呼び出しが1回しかない」典型的な穴。")
    PAIRS = [
        ("sire_stats_father.csv", "sire_stats_father_train.csv", "種牡馬成績(父)"),
        ("sire_stats_bms.csv", "sire_stats_bms_train.csv", "種牡馬成績(母父)"),
        ("sire_stats.csv", "sire_stats_train.csv", "種牡馬成績(統合)"),
    ]
    for a, b, desc in PAIRS:
        ap, bp = os.path.join(BASE, a), os.path.join(BASE, b)
        if not (os.path.exists(ap) and os.path.exists(bp)):
            log(f"  {desc:<20}― どちらか無い")
            continue
        d = abs(os.path.getmtime(ap) - os.path.getmtime(bp)) / 86400
        ok = d <= 1.0
        log(f"  {desc:<20}差 {d:>5.1f}日  {'○' if ok else '⚠ 片方だけ古い'}")
        if not ok:
            ng("高", f"{desc}: {a} と {b} の更新が{d:.1f}日ずれている。"
                     f"片方しか作られていない可能性")

    log("\n" + "=" * 70)
    log("  ③ 設定の二重定義")
    log("=" * 70)
    CONST = {
        "AX_GAP(軸のしきい値)": [("resid_io.py", r"^AX_GAP\s*=\s*([\d.]+)"),
                             ("flask_app.py", r"^RESID_AX\s*=\s*([\d.]+)")],
        "MATE_GAP(相手のしきい値)": [("resid_io.py", r"^MATE_GAP\s*=\s*([\d.]+)"),
                                ("flask_app.py", r"^RESID_MATE\s*=\s*([\d.]+)")],
        "GRADE_TH(評価S)": [("flask_app.py", r'\("S",\s*([\d.]+)\)'),
                          ("resid_sgrade.py", r'\("S",\s*([\d.]+)\)')],
    }
    for name, srcs in CONST.items():
        vals = {}
        for f, pat in srcs:
            p = os.path.join(BASE, f)
            if not os.path.exists(p):
                continue
            m = re.search(pat, open(p, encoding="utf-8").read(), re.M)
            if m:
                vals[f] = m.group(1)
        if len(set(vals.values())) > 1:
            log(f"  ⚠ {name}: " + " / ".join(f"{k}={v}" for k, v in vals.items()))
            ng("高", f"{name} が食い違っている: {vals}")
        elif vals:
            log(f"  ○ {name}: {list(vals.values())[0]}（{len(vals)}か所で一致）")

    log("\n" + "=" * 70)
    log("  ③b 学習と本番で同じ入力を使っているか")
    log("=" * 70)
    log("  モデルは学習時の値の分布で木を作る。本番だけ違う値を渡すと判断が狂う。")
    log("  2026-08-18に実際に起きた: 本番だけ全期間の血統を使い、軸に選ぶ馬が")
    log("  15.1%のレースで変わっていた（gapの相関 0.941）。")
    try:
        txt = open(os.path.join(BASE, "features.py"), encoding="utf-8").read()
        # 呼び出しだけを見る。関数定義の既定値（def ...=False）は対象外。
        # 定義まで拾うと「不一致」と誤検知する（2026-08-18に実際に誤検知した）。
        tr = [m[1] if isinstance(m, tuple) else m for m in
              re.findall(r"(?<!def )_run_feature_pipeline\([^)]*"
                         r"use_train_snapshot\s*=\s*(True|False)", txt)]
        log(f"  features.py の血統スナップショット指定: {tr}")
        if len(set(tr)) > 1:
            log("  ⚠ 学習と本番で指定が違う")
            ng("高", f"features.py の use_train_snapshot が不一致: {tr}。"
                     f"学習と本番で違う血統値がモデルに渡る")
        elif tr:
            log(f"  ○ すべて {tr[0]} で揃っている")
    except Exception as e:
        log(f"  確認できず: {type(e).__name__}")

    log("\n" + "=" * 70)
    log("  ④ 飛ばされている工程")
    log("=" * 70)
    for f in ("weekly_update.py", "auto_predict_publish.py"):
        p = os.path.join(BASE, f)
        if not os.path.exists(p):
            continue
        txt = open(p, encoding="utf-8").read()
        sk = re.findall(r'"--skip-([a-z-]+)"', txt)
        if sk:
            log(f"  {f}: --skip-{' / --skip-'.join(sorted(set(sk)))}")
            for s in set(sk):
                log(f"    → 「{s}」は別工程で実施されているか要確認")
        else:
            log(f"  {f}: skip指定なし")

    log("\n" + "=" * 70)
    log("  ⑤ 週次工程の網羅性")
    log("=" * 70)
    wp = os.path.join(BASE, "weekly_update.py")
    if os.path.exists(wp):
        txt = open(wp, encoding="utf-8").read()
        need = ["update_data.py", "horse_scraper.py", "sire_stats.py", "features.py",
                "train_mf_v2.py", "train_resid.py", "build_calibrator.py",
                "result_tracker.py", "paper_report.py", "build_grade.py"]
        for n in need:
            inc = n in txt
            log(f"  {'○' if inc else '⚠'} {n:<24}{'週次に含まれる' if inc else '週次に無い'}")
            if not inc:
                ng("中", f"{n} が週次更新に入っていない")

    log("\n" + "=" * 70)
    log("  ⑤a 自動投票のガード（お金が動く経路）")
    log("=" * 70)
    log("  1つでも開いていれば実弾が出る可能性がある。毎回ここを確認する。")
    try:
        kp = open(os.path.join(BASE, "keiba_predict.py"), encoding="utf-8").read()
        av = open(os.path.join(BASE, "auto_vote.py"), encoding="utf-8").read()
        be = re.search(r"^BETTING_ENABLED\s*=\s*(True|False)", kp, re.M)
        vm = re.search(r'^VOTE_MODE\s*=\s*"(\w+)"', av, re.M)
        armed = os.path.exists(os.path.join(BASE, "AUTO_VOTE_ARMED"))
        env = os.path.join(BASE, ".env")
        ipat = ("IPAT_" in open(env, encoding="utf-8").read()) if os.path.exists(env) else False
        # 投票する対象があるか。取得元(BET_SOURCE)によって見るファイルが違う。
        # resid = paper_resid.csv の判定「買い」/ legacy = today_bets.csv
        bs = re.search(r'^BET_SOURCE\s*=\s*"(\w+)"', av, re.M)
        src = bs.group(1) if bs else "?"
        # BET_SOURCE="resid" では paper_resid.csv（＝前向き検証の記録。購入停止中でも
        # 「買い」行が書かれる）を読むので、その存在はガードにならない。
        # 代わりに auto_vote._effective_mode() の実測を最後の一段として見る。
        guards = [
            ("BETTING_ENABLED", be.group(1) if be else "?", "False"),
            ("VOTE_MODE", vm.group(1) if vm else "?", "dryrun"),
            ("AUTO_VOTE_ARMED", "あり" if armed else "なし", "なし"),
            ("IPAT認証", "設定済" if ipat else "未設定", "未設定"),
        ]
        log(f"  買い目の取得元: BET_SOURCE={src}"
            f"（{'残差モデル' if src == 'resid' else '旧方式' if src == 'legacy' else '不明'}）")
        if src not in ("resid", "legacy"):
            ng("高", f"auto_vote.BET_SOURCE が不明な値({src})。投票対象が読めない")
        elif src == "legacy":
            ng("高", "auto_vote.BET_SOURCE=legacy。購入停止した旧方式を投票する設定になっている")
        opened = 0
        for name, cur, safe in guards:
            ok = (cur == safe)
            opened += 0 if ok else 1
            log(f"  {'○ 閉' if ok else '⚠ 開'} {name:<20}{cur}")
        # 最後に、実際に効くモードを auto_vote 自身に聞く（設定の読み違えを防ぐ）
        try:
            import auto_vote as _av
            eff, why = _av._effective_mode()
            log(f"  {'○ 閉' if eff == 'dryrun' else '⚠ 開'} {'実効モード':<20}{eff}"
                + (f"（{why}）" if why else ""))
            if eff != "dryrun":
                opened += 1
                ng("高", "auto_vote が実投票モードで動く状態にある")
        except Exception as e:
            log(f"  ⚠ 実効モードを確認できず: {type(e).__name__}")
            ng("中", "auto_vote._effective_mode() が読めない")

        if opened == 0:
            log("  → 全段閉じている。お金は動かない")
        else:
            log(f"  → {opened}段が開いている。意図した切り替えか確認すること")
            ng("高" if opened >= 3 else "中",
               f"自動投票のガードが{opened}段開いている。AUTO_VOTE_手順書.md を参照")
    except Exception as e:
        log(f"  確認できず: {type(e).__name__}")

    log("\n" + "=" * 70)
    log("  ⑤b 集計ツールが実際に動くか")
    log("=" * 70)
    log("  paper_report.py が KeyError で落ちていたのに気づかなかった（2026-08-22）。")
    log("  週次で呼んでいても、落ちていれば何も分からない。実際に動かして確かめる。")
    import subprocess as _sp
    # check_resid.py を足した理由（2026-08-25）
    #   8/17にしきい値を2.0→1.5に緩めたとき EXPECT の更新を忘れ、以後8日間
    #   「⚠ 実装がズレている」と誤警告を出し続けていた。誰も走らせていないので
    #   気づけなかった。オオカミ少年のまま放置すると、本当にズレたときに効かない。
    for _t in ("paper_report.py", "audit_calls.py", "check_resid.py"):
        _p = os.path.join(BASE, _t)
        if not os.path.exists(_p):
            continue
        try:
            _r = _sp.run(["python", _p], cwd=BASE, capture_output=True,
                         text=True, timeout=300,
                         env=dict(os.environ, PYTHONUTF8="1"))
            _ok = _r.returncode == 0
            log(f"  {'○' if _ok else '⚠'} {_t:<22}"
                f"{'正常終了' if _ok else '異常終了'}")
            if not _ok:
                _last = [x for x in (_r.stderr or "").splitlines() if x.strip()]
                log(f"      {_last[-1][:80] if _last else ''}")
                ng("高", f"{_t} が異常終了する。集計できない状態")
        except Exception as e:
            log(f"  ⚠ {_t:<22}実行できず（{type(e).__name__}）")
            ng("中", f"{_t} を実行できない")

    log("\n" + "=" * 70)
    log("  ⑥ 参照切れ（コードが読むファイルが存在するか）")
    log("=" * 70)
    MUST = ["market_free_model.py", "resid_io.py", "model_diag.py", "features.py",
            "grade_calib.pkl", "model_resid.pkl", "jv_payouts.csv",
            "sire_stats_father_train.csv", "course_bias.csv"]
    for f in MUST:
        e = os.path.exists(os.path.join(BASE, f))
        log(f"  {'○' if e else '⚠'} {f}")
        if not e:
            ng("高", f"{f} が存在しない")

    log("\n" + "=" * 70)
    log("  まとめ")
    log("=" * 70)
    if not NG:
        log("  ✅ 異常なし")
        return
    for sev in ("高", "中", "低"):
        xs = [m for s, m in NG if s == sev]
        if xs:
            log(f"\n  【重要度{sev}】{len(xs)}件")
            for m in xs:
                log(f"    - {m}")


if __name__ == "__main__":
    main()
