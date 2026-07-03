"""
retrain_full.py
────────────────
horse_scraper.py 完走後に手動実行するスクリプト。
血統データを完全反映してモデルを再学習する。

実行タイミング:
  horse_master.csv が 42,000頭以上になったことを確認してから実行。

使い方:
    python retrain_full.py
"""

import os
import sys
import subprocess
from datetime import datetime

BASE_DIR = r"c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai"
PYTHON   = r"C:/Users/別府飛河/AppData/Local/Microsoft/WindowsApps/python3.11.exe"
LOG_FILE = os.path.join(BASE_DIR, "retrain_log.txt")


def log(msg):
    ts = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(label, cmd, timeout=7200):
    log(f"--- {label} 開始 ---")
    try:
        result = subprocess.run(
            cmd, cwd=BASE_DIR, capture_output=True,
            text=True, encoding="utf-8", timeout=timeout
        )
        for line in (result.stdout or "").strip().splitlines()[-15:]:
            log(f"  {line}")
        if result.returncode != 0:
            log(f"  [警告] 終了コード {result.returncode}")
            for line in (result.stderr or "").strip().splitlines()[-5:]:
                log(f"  ERR: {line}")
            return False
        log(f"--- {label} 完了 ---")
        return True
    except subprocess.TimeoutExpired:
        log(f"  [タイムアウト] {label} が {timeout//3600}時間を超過")
        return False
    except Exception as e:
        log(f"  [エラー] {label}: {e}")
        return False


def check_horse_master():
    """horse_master.csv の取得頭数を確認"""
    import pandas as pd
    hm_path = os.path.join(BASE_DIR, "horse_master.csv")
    if not os.path.exists(hm_path):
        log("horse_master.csv が存在しません")
        return False
    hm = pd.read_csv(hm_path)
    n = len(hm)
    log(f"horse_master.csv: {n:,}頭")
    if n < 40000:
        log(f"  [警告] まだ取得中の可能性があります（目標: 42,046頭）")
        ans = input("このまま続行しますか？ (y/n): ").strip().lower()
        return ans == "y"
    return True


def main():
    log("=" * 50)
    log("血統データ完全反映 再学習 開始")
    log("=" * 50)

    # 前提チェック
    if not check_horse_master():
        log("中断しました")
        sys.exit(1)

    start = datetime.now()

    # Step 1: 特徴量再生成（血統データを race_features.csv に反映）
    ok = run_step(
        "特徴量再生成（血統完全版）",
        [PYTHON, "-c",
         "import sys; sys.path.insert(0, r'" + BASE_DIR + "'); "
         "from features import build_features; build_features()"],
        timeout=7200  # 2時間
    )
    if not ok:
        log("特徴量再生成に失敗しました。中断します。")
        sys.exit(1)

    # Step 2: 通常モデル再学習
    run_step(
        "通常モデル再学習",
        [PYTHON, os.path.join(BASE_DIR, "model.py")],
        timeout=3600
    )

    # Step 3: MFモデル再学習
    run_step(
        "MFモデル再学習",
        [PYTHON, os.path.join(BASE_DIR, "market_free_model.py")],
        timeout=3600
    )

    # Step 4: GitHub push
    log("--- GitHub push ---")
    try:
        push_files = [
            "race_features.csv",
            "feature_importance.csv",
            "model_result.csv",
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
                ["git", "commit", "-m", f"血統データ完全反映・モデル再学習 {date_str}"],
                cwd=BASE_DIR, capture_output=True
            )
            subprocess.run(["git", "push"], cwd=BASE_DIR, capture_output=True)
            log("  GitHub push 完了")
        else:
            log("  変更なし・スキップ")
    except Exception as e:
        log(f"  Git エラー: {e}")

    elapsed = datetime.now() - start
    log("=" * 50)
    log(f"完了  所要時間: {elapsed}")
    log("=" * 50)
    log("")
    log("【次のステップ】")
    log("  1. ダッシュボード「📈 精度分析」で血統特徴量の重要度上昇を確認")
    log("  2. ◎の1番人気率が改善しているか確認（目標: 70%未満）")
    log("  3. 戦略判定のMF統一（バックテスト再実行とセット）を検討")


if __name__ == "__main__":
    main()
