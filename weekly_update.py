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
from datetime import datetime

BASE_DIR = r"c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai"
PYTHON   = r"C:/Users/別府飛河/AppData/Local/Microsoft/WindowsApps/python3.11.exe"
LOG_FILE = os.path.join(BASE_DIR, "weekly_update_log.txt")


def log(msg):
    ts = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(label, cmd, timeout=3600):
    """サブプロセスを実行してログに記録。失敗してもクラッシュしない。"""
    log(f"--- {label} 開始 ---")
    try:
        result = subprocess.run(
            cmd, cwd=BASE_DIR, capture_output=True,
            text=True, encoding="utf-8", timeout=timeout
        )
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines()[-10:]:  # 末尾10行だけログ
                log(f"  {line}")
        if result.returncode != 0:
            log(f"  [警告] 終了コード {result.returncode}")
            if result.stderr.strip():
                for line in result.stderr.strip().splitlines()[-5:]:
                    log(f"  ERR: {line}")
        else:
            log(f"--- {label} 完了 ---")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log(f"  [タイムアウト] {label} が {timeout}秒 を超過しました")
        return False
    except Exception as e:
        log(f"  [エラー] {label}: {e}")
        return False


def main():
    log("=" * 50)
    log("週次自動更新 開始")
    log("=" * 50)

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

    # ── Step 2: 通常モデル再学習 ──────────────────────────────────────────
    log("[Step2] モデル再学習（通常モデル）")
    run_step(
        "model.py",
        [PYTHON, os.path.join(BASE_DIR, "model.py")],
        timeout=1800  # 30分
    )

    # ── Step 3: MFモデル再学習 ────────────────────────────────────────────
    log("[Step3] MFモデル再学習")
    run_step(
        "market_free_model.py",
        [PYTHON, os.path.join(BASE_DIR, "market_free_model.py")],
        timeout=1800  # 30分
    )

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


if __name__ == "__main__":
    main()
