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
