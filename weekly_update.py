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

    # ── Step 5: GitHubプッシュ ────────────────────────────────────────────
    log("[Step5] GitHubプッシュ")
    try:
        push_files = [
            "prediction_record_v2.csv",
            "model_result.csv",
            "weekly_update_log.txt",
            "accuracy_log.csv",
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
