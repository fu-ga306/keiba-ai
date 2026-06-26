# -*- coding: utf-8 -*-
"""
auto_pipeline.py
スクレイピング完了を自動検知し、ロードマップ A-4〜C-3 を順番に実行する:

  [A-4] cleaner.py           -> race_data_clean.csv  +欠損率確認
  [C-1] 血統結合率チェック   -> horse_id結合率レポート
  [C-2] sire_stats.py        -> sire_stats_father.csv / sire_stats_bms.csv
  [B-3/C-3] features.py     -> race_features.csv (回り_num・賞金・血統 を含む)
  [B-1] model.py             -> model.pkl / model_result.csv
  [B-1] market_free_model.py -> model_mf.pkl / model_mf_result.csv
  [B-3] analyze_feature_importance.py (MF・通常モデル両方)
  [B-2] analyze_mf_roi.py   -> 回収率レポート
"""
import subprocess
import sys
import time
import os
import csv

PY   = sys.executable
BASE = os.path.dirname(os.path.abspath(__file__))

SCRAPER_DONE_SIGNAL = os.path.join(BASE, "race_data_large.csv")
SCRAPER_LOG         = os.path.join(BASE, "scraper_log.txt")


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── スクレイピング完了待ち ────────────────────────────────────────────
def wait_for_scraper():
    log("スクレイピング完了を監視中（1分ごとにチェック）...")
    log(f"  完了シグナル: race_data_large.csv の出現")
    check = 0
    while True:
        if os.path.exists(SCRAPER_DONE_SIGNAL):
            log("race_data_large.csv を検出 -> スクレイピング完了！")
            time.sleep(3)
            return
        check += 1
        print(f"  [{check}分経過] まだ実行中...", flush=True)
        time.sleep(60)


# ── スクリプト実行 ────────────────────────────────────────────────────
def run_step(script_name, args=None):
    sep = "=" * 60
    log(f"\n{sep}")
    log(f"実行中: {script_name}")
    log(sep)
    start = time.time()

    cmd = [PY, os.path.join(BASE, script_name)]
    if args:
        cmd += args

    result = subprocess.run(cmd, cwd=BASE)
    elapsed = time.time() - start

    if result.returncode != 0:
        log(f"[ERROR] {script_name} が異常終了 (code={result.returncode})")
        log("パイプラインを中止します。")
        sys.exit(1)

    log(f"[完了] {script_name}  ({elapsed/60:.1f}分)")


# ── C-1: 血統結合率チェック ───────────────────────────────────────────
def check_bloodline_join_rate():
    sep = "=" * 60
    log(f"\n{sep}")
    log("[C-1] 血統データ結合率チェック")
    log(sep)

    clean_path  = os.path.join(BASE, "race_data_clean.csv")
    master_path = os.path.join(BASE, "horse_master.csv")

    if not os.path.exists(clean_path):
        log("  race_data_clean.csv が見つかりません")
        return
    if not os.path.exists(master_path):
        log("  horse_master.csv が見つかりません -> 血統結合スキップ")
        return

    try:
        import pandas as pd

        clean  = pd.read_csv(clean_path, usecols=["horse_id"], low_memory=False)
        master = pd.read_csv(master_path, low_memory=False)

        total = len(clean)
        has_id = clean["horse_id"].notna().sum()
        id_rate = has_id / total * 100

        # horse_masterとのjoin
        if "horse_id" in master.columns:
            master["horse_id"] = master["horse_id"].astype(str).str.replace(r"\.0$", "", regex=True)
            clean["horse_id_str"] = clean["horse_id"].astype(str).str.replace(r"\.0$", "", regex=True)
            matched = clean["horse_id_str"].isin(master["horse_id"].astype(str)).sum()
            match_rate = matched / total * 100
        else:
            match_rate = 0.0

        log(f"  horse_id あり: {has_id:,}/{total:,}行 ({id_rate:.1f}%)")
        log(f"  horse_master 結合率: {match_rate:.1f}%  (再スクレイピング前: 66.9%)")
        if match_rate > 80:
            log("  -> 結合率80%超: 血統特徴量が本来の効果を発揮できる水準に達しました")
        elif match_rate > 66.9:
            log(f"  -> 旧66.9%から改善 ({match_rate:.1f}%): 血統効果が上昇している可能性あり")
        else:
            log(f"  -> 結合率が低いまま ({match_rate:.1f}%): horse_scraper.py の追加実行を検討")
    except Exception as e:
        log(f"  チェック中にエラー: {e}")


# ── 結果サマリ表示 ────────────────────────────────────────────────────
def show_result_summary():
    sep = "=" * 60
    log(f"\n{sep}")
    log("結果ファイルサマリ")
    log(sep)

    files = [
        ("race_data_clean.csv",    "前処理済みデータ"),
        ("race_features.csv",      "特徴量データ"),
        ("model.pkl",              "通常モデル"),
        ("model_result.csv",       "通常モデル予測結果"),
        ("model_mf.pkl",           "MFモデル"),
        ("model_mf_result.csv",    "MFモデル予測結果"),
        ("feature_importance.csv", "特徴量重要度"),
    ]
    for fname, label in files:
        path = os.path.join(BASE, fname)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            log(f"  {label} ({fname}): {size_mb:.1f}MB")
        else:
            log(f"  {label} ({fname}): 未生成")


# ── メイン ────────────────────────────────────────────────────────────
def main():
    os.chdir(BASE)
    log("=== 自動パイプライン開始 (ロードマップ A-4〜C-3) ===")

    # A-3 完了待ち
    wait_for_scraper()

    # A-4: 前処理
    run_step("cleaner.py")

    # C-1: 血統結合率チェック
    check_bloodline_join_rate()

    # C-2: 血統統計再生成（結合率が上がった状態でsire_statsを作り直す）
    run_step("sire_stats.py")

    # B-3/C-3: 特徴量生成（回り_num・賞金・血統を含む）
    run_step("features.py")

    # B-1: 通常モデル学習
    run_step("model.py")

    # B-1: MFモデル学習
    run_step("market_free_model.py")

    # B-3: 特徴量重要度分析（MFモデル）
    log("\n-- 特徴量重要度分析: MFモデル --")
    run_step("analyze_feature_importance.py")

    # B-3: 特徴量重要度分析（通常モデル）
    log("\n-- 特徴量重要度分析: 通常モデル --")
    run_step("analyze_feature_importance.py", args=["normal"])

    # B-2: 回収率分析
    run_step("analyze_mf_roi.py")

    # 結果サマリ
    show_result_summary()

    log("\n=== 全パイプライン完了 (A-4〜C-3) ===")
    log("次のステップ:")
    log("  C-4: analyze_mf_roi.py の結果を見て MFモデル主力化を判断")
    log("  D:   信号分類・荒れスコア・UIの実装（要別途指示）")


if __name__ == "__main__":
    main()
