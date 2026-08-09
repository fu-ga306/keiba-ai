"""
weekly_update.py
────────────────
毎週月曜日 08:00 に自動実行するスクリプト。

実行順序:
  0. レース結果スクレイピング（直近2週分）→ race_data_clean.csv 更新
  1. クリーニング（cleaner.py）
  2. sire_stats 再集計
  3. 特徴量再生成（features.py）
  4. モデル再学習（model.py + market_free_model.py）
  5. result_tracker.py update（予想記録の照合）
  6. GitHubにプッシュ（予想記録・モデル結果）

タスクスケジューラ登録済み：毎週月曜 08:00
"""

import os
import sys
import subprocess
from datetime import datetime, timedelta

BASE_DIR = r"c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai"
PYTHON   = r"C:/Users/別府飛河/AppData/Local/Microsoft/WindowsApps/python3.11.exe"
LOG_FILE = os.path.join(BASE_DIR, "weekly_update_log.txt")


def log(msg):
    ts = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


STEP_RESULTS = []   # (ラベル, 成否, 所要秒) を貯めて最後にメールで知らせる


def run_step(label, cmd, timeout=3600):
    """サブプロセスを実行してログに記録。失敗してもクラッシュしない。"""
    log(f"--- {label} 開始 ---")
    _t0 = datetime.now()
    try:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"  # Windows CP932 → UTF-8 強制（エンコードエラー防止）
        result = subprocess.run(
            cmd, cwd=BASE_DIR, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout,
            env=env
        )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if stdout:
            for line in stdout.splitlines()[-10:]:
                log(f"  {line}")
        if result.returncode != 0:
            log(f"  [警告] 終了コード {result.returncode}")
            if stderr:
                for line in stderr.splitlines()[-5:]:
                    log(f"  ERR: {line}")
        else:
            log(f"--- {label} 完了 ---")
        STEP_RESULTS.append((label, result.returncode == 0,
                             int((datetime.now() - _t0).total_seconds())))
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log(f"  [タイムアウト] {label} が {timeout}秒 を超過しました")
        STEP_RESULTS.append((label + "（時間切れ）", False,
                             int((datetime.now() - _t0).total_seconds())))
        return False
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log(f"  [エラー] {label}: {e}")
        for line in tb.splitlines()[-10:]:
            log(f"  TB: {line}")
        STEP_RESULTS.append((label + "（例外）", False,
                             int((datetime.now() - _t0).total_seconds())))
        return False


def _is_race_day():
    """今日が開催日か。ファイルの鮮度だけで判断し、スクレイピングはしない。

    週次は火曜に移したが、山の日などの祝日火曜には開催があり得る。
    開催日に走らせると Step0 のスクレイピングが当日の予想・結果取得と
    ぶつかり、ブロックを招く（2026-07-27に実際に400を食らっている）。
    """
    today = datetime.now().date()
    for fn in ("today_race_times.json", "today_predictions.csv"):
        p = os.path.join(BASE_DIR, fn)
        if os.path.exists(p) and \
                datetime.fromtimestamp(os.path.getmtime(p)).date() == today:
            return True
    return False


def main():
    log("=" * 50)
    log("週次自動更新 開始")
    log("=" * 50)

    # 開催日には走らせない（2026-08-09）。Step0は直近2週を取り直すので、
    # 1週飛ばしてもデータは失われない。翌週の実行で追いつく。
    if _is_race_day():
        log("本日は開催日 → 週次更新を見送ります（来週に実施）")
        STEP_RESULTS.append(("開催日のため見送り", True, 0))
        _notify(timedelta(0))
        return

    start = datetime.now()

    # ── Step 0: スクレイピング（直近2週分の新規レース取得） ───────────────
    log("[Step0] レース結果スクレイピング（直近2週）")
    run_step(
        "スクレイピング",
        [PYTHON, os.path.join(BASE_DIR, "update_data.py"),
         "--weeks", "2",
         "--skip-horse",      # horse_scraperは時間がかかるため別タスクで実施
         "--skip-features",   # features/modelは後ほど個別実行
         "--skip-model"],
        timeout=1800  # 30分
    )

    # ── Step 1: 特徴量再生成 ───────────────────────────────────────────────
    log("[Step1] 特徴量再生成")
    run_step(
        "features.py",
        [PYTHON, "-c",
         "import sys; sys.path.insert(0, r'" + BASE_DIR + "'); "
         "from features import build_features; "
         "build_features()"],
        timeout=3600  # 60分
    )

    # ── モデル固定モード（2026-08-09）──────────────────────────────────
    #   FREEZE_MODEL というファイルを置くと、学習をやり直す工程(Step2/3/3.5/3.6)
    #   を飛ばす。予想は今の model.pkl / model_mf.pkl / mf_calibrator.pkl で
    #   出し続ける。
    #
    #   なぜ用意したか: 印の成績や評価の精度を数ヶ月かけて検証したいのに、
    #   毎週モデルが変わると「成績が変わったのはモデル更新のせいか偶然か」を
    #   切り分けられない。学習物を固定すれば、貯まったデータは全て同じモデルの
    #   出力になり、そのまま統計にかけられる。
    #
    #   飛ばすのは学習だけで、Step0(結果の取得)とStep1(特徴量・各種集計の更新)は
    #   続ける。新しいレース結果は貯まり続けるので、検証が終わってから
    #   このファイルを消せば、その時点の全データで学習し直せる。
    frozen = os.path.exists(os.path.join(BASE_DIR, "FREEZE_MODEL"))
    if frozen:
        log("[モデル固定] FREEZE_MODEL があるため Step2/3/3.5/3.6 を飛ばします")
        log("            （予想は今の学習物のまま。結果の蓄積と特徴量更新は続行）")
        STEP_RESULTS.append(("Step2〜3.6 学習（固定モードのため実施せず）", True, 0))

    # ── Step 2: 通常モデル再学習（本番モード=全データ学習 → model.pkl）───────
    #   2026-07-16: timeout 30分→4時間（7/13に30分超過で毎週失敗していたのを修正）。
    #   backtest資産(model_result*.csv)は本番モードでは更新されない（model.py backtestで別途）。
    log("[Step2] モデル再学習（通常モデル・全データ）")
    if not frozen:
      run_step(
        "model.py",
        [PYTHON, os.path.join(BASE_DIR, "model.py")],
        timeout=14400  # 4時間
      )

    # ── Step 3: MFモデル再学習（改善版v2・本番モード=全データ → model_mf.pkl）──
    #   2026-07-16: 旧market_free_model.py→train_mf_v2.py（バギング+LambdaRank版）に切替。
    log("[Step3] MFモデル再学習（train_mf_v2）")
    if not frozen:
      run_step(
        "train_mf_v2.py",
        [PYTHON, os.path.join(BASE_DIR, "train_mf_v2.py")],
        # 2026-08-03に7200秒で足りずタイムアウトした（他の処理と並走していた影響もある）。
        # 翌日は開催がないので余裕を持たせる。
        timeout=21600  # 6時間
      )

    # ── Step 3.5: 確率較正用のOOS出力を更新 ──────────────────────────────
    #   買い判定は「較正済み勝率 × 実オッズ」の期待値で行う（2026-08-04導入）。
    #   MFは正例重み(win 2.0/place3 1.5)と時間重みで学習しているため生の確率は
    #   過大で、2025 OOSでは 勝率 予測11.0%→実際7.2%（1.5倍）だった。
    #   較正しないと期待値も推奨賭け率もずれ、買う馬自体が変わる。
    #   較正器は「正直なOOS出力」から作る必要があり、backtestモード
    #   （≤前年学習/当年検証）でのみ model_mf_result.csv が更新される。
    #   本番モデル(model_mf.pkl)には触らない（出力は model_mf_bt.pkl）。
    log("[Step3.5] 較正用のOOS出力を更新（train_mf_v2 backtest）")
    ok_bt = False if frozen else run_step(
        "train_mf_v2.py backtest",
        [PYTHON, os.path.join(BASE_DIR, "train_mf_v2.py"), "backtest"],
        timeout=21600  # 6時間
    )

    # ── Step 3.6: 確率較正器を作り直す ──────────────────────────────────
    #   モデルが変われば確率の癖も変わるので、毎週作り直す。
    #   失敗しても keiba_predict は較正器が無ければ生の確率で動く（予測は止まらない）
    #   が、その場合は期待値が過大になり買い過ぎるため、警告を残す。
    if ok_bt:
        log("[Step3.6] 確率較正器を再作成")
        if not run_step("build_calibrator.py",
                        [PYTHON, os.path.join(BASE_DIR, "build_calibrator.py")],
                        timeout=1800):
            log("  [警告] 較正器の再作成に失敗。古い較正器のまま動きます")
    elif frozen:
        log("[Step3.6] スキップ（モデル固定モード。較正器も今のまま使います）")
    else:
        log("[Step3.6] スキップ（Step3.5が失敗したため較正器は更新しない）")
        log("  [警告] 較正器が古いままです。期待値がずれる可能性があります")

    # ── Step 4: result_tracker 更新 ───────────────────────────────────────
    log("[Step4] レース結果照合")
    run_step(
        "result_tracker update",
        [PYTHON, os.path.join(BASE_DIR, "result_tracker.py"), "update"],
        timeout=300
    )
    run_step(
        "result_tracker summary",
        [PYTHON, os.path.join(BASE_DIR, "result_tracker.py"), "summary"],
        timeout=120
    )

    # ── Step 4.5: 精度サマリー自動生成 ───────────────────────────────────
    log("[Step4.5] 精度サマリー生成")
    try:
        import pandas as pd
        import numpy as np

        rec_path = os.path.join(BASE_DIR, "prediction_record_v2.csv")
        df = pd.read_csv(rec_path)
        confirmed = df[df["hit"].notna()].copy()
        confirmed["honmei_actual"] = pd.to_numeric(confirmed["honmei_actual"], errors="coerce")
        confirmed["honmei_ninki"]  = pd.to_numeric(confirmed["honmei_ninki"], errors="coerce")
        confirmed["honmei_ev"]     = pd.to_numeric(confirmed["honmei_ev"], errors="coerce")

        n = len(confirmed)
        overall_win  = (confirmed["honmei_actual"] == 1).mean() * 100
        overall_fuku = (confirmed["honmei_actual"] <= 3).mean() * 100

        by_jyo = confirmed.groupby("jyo").agg(
            件数=("honmei_actual", "count"),
            勝率=("honmei_actual", lambda x: (x==1).mean()*100),
        ).reset_index()

        issues = []
        if overall_win < 25:
            issues.append(f"◎勝率低下({overall_win:.1f}%) → コース形態/血統特徴量の効果確認")
        weak = by_jyo[by_jyo["勝率"] < 20]
        if not weak.empty:
            issues.append(f"弱点競馬場: {', '.join(weak['jyo'].tolist())} → 直線長_m・坂あり要チェック")
        nb1_pct = (confirmed["honmei_ninki"] == 1).mean() * 100
        if nb1_pct > 70:
            issues.append(f"◎の{nb1_pct:.0f}%が1番人気 → MF識別力不足、B-1血統データ追加を検討")
        ev_pos = confirmed[confirmed["honmei_ev"] >= 0]
        ev_neg = confirmed[confirmed["honmei_ev"] < 0]
        if len(ev_pos) > 5 and len(ev_neg) > 5:
            wp  = (ev_pos["honmei_actual"] == 1).mean() * 100
            wn  = (ev_neg["honmei_actual"] == 1).mean() * 100
            if wp < wn:
                issues.append(f"EV≥0の勝率({wp:.1f}%) < EV<0({wn:.1f}%) → EVフィルター閾値再検討")

        summary_path = os.path.join(BASE_DIR, "accuracy_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"=== 精度分析サマリー {datetime.now().strftime('%Y/%m/%d')} ===\n\n")
            f.write(f"◎勝率: {overall_win:.1f}%  複勝率: {overall_fuku:.1f}%  照合件数: {n}件\n\n")
            f.write("【競馬場別】\n")
            for _, row in by_jyo.iterrows():
                f.write(f"  {row['jyo']}: {row['勝率']:.1f}% ({row['件数']}件)\n")
            f.write("\n【月曜改修ポイント】\n")
            for issue in (issues or ["現時点で明確な弱点なし"]):
                f.write(f"- {issue}\n")
        log(f"  精度サマリー保存: {summary_path}")
    except Exception as e:
        log(f"  精度サマリー生成エラー: {e}")

    # ── Step 5: GitHubプッシュ ────────────────────────────────────────────
    log("[Step5] GitHubプッシュ")
    try:
        push_files = [
            "prediction_record_v2.csv",
            "model_result.csv",
            "weekly_update_log.txt",
            "accuracy_log.csv",
            "accuracy_summary.txt",
            # 確率較正器（2026-08-05追加・7KB）。買い判定が較正済み確率に
            # 依存するようになったため、これが古いと期待値がずれる。
            "mf_calibrator.pkl",
        ]
        for f in push_files:
            fpath = os.path.join(BASE_DIR, f)
            if os.path.exists(fpath):
                subprocess.run(["git", "add", fpath], cwd=BASE_DIR, capture_output=True)

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=BASE_DIR, capture_output=True, text=True
        )
        if status.stdout.strip():
            date_str = datetime.now().strftime("%Y/%m/%d")
            subprocess.run(
                ["git", "commit", "-m", f"週次自動更新 {date_str}"],
                cwd=BASE_DIR, capture_output=True
            )
            subprocess.run(["git", "push"], cwd=BASE_DIR, capture_output=True)
            log("  GitHubプッシュ完了")
        else:
            log("  変更なし・スキップ")
    except Exception as e:
        log(f"  Gitエラー: {e}")

    elapsed = datetime.now() - start
    log("=" * 50)
    log(f"週次自動更新 完了  所要時間: {elapsed}")
    log("=" * 50)
    _notify(elapsed)


def _notify(elapsed):
    """結果をメールで知らせる（2026-08-09追加）。

    これまでは weekly_update_log.txt に書くだけだったので、8/3にStep3が
    時間切れでMFモデルが更新されないまま終わっていたことに1週間気づけなかった。
    無人運用では、失敗が届かないことが一番こわい。
    """
    ng = [l for l, ok, _ in STEP_RESULTS if not ok]
    head = "✅週次更新 完了" if not ng else f"⚠週次更新 {len(ng)}件が失敗"
    lines = [f"所要時間 {elapsed}", ""]
    for l, ok, sec in STEP_RESULTS:
        lines.append(f"  {'○' if ok else '×'} {l}  {sec // 60}分")
    lines.append("")
    lines.append("モデルファイルの更新日時:")
    for f in ("model.pkl", "model_mf.pkl", "mf_calibrator.pkl", "race_features.csv"):
        fp = os.path.join(BASE_DIR, f)
        if os.path.exists(fp):
            m = datetime.fromtimestamp(os.path.getmtime(fp))
            age = (datetime.now() - m).days
            lines.append(f"  {f}: {m:%m/%d %H:%M}" + (f"  ← {age}日前のまま" if age >= 6 else ""))
        else:
            lines.append(f"  {f}: 見つからない")
    if ng:
        lines += ["", "失敗したもの:"] + [f"  {l}" for l in ng]
        lines += ["", "詳しくは weekly_update_log.txt を見てください。"]
    body = chr(10).join(lines)
    log(body)
    try:
        import smtplib
        from email.mime.text import MIMEText
        from dotenv import load_dotenv
        load_dotenv(os.path.join(BASE_DIR, ".env"))
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = f"{head}（{datetime.now():%m/%d}）"
        msg["From"] = os.environ["GMAIL_ADDRESS"]
        msg["To"] = os.environ["TO_ADDRESS"]
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(os.environ["GMAIL_ADDRESS"], os.environ["GMAIL_APP_PASS"])
            srv.send_message(msg)
        log("  完了メールを送信しました")
    except Exception as e:
        log(f"  完了メールの送信に失敗: {e}")


if __name__ == "__main__":
    main()
