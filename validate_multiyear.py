# -*- coding: utf-8 -*-
"""多年度out-of-sample検証: <=(TY-1)学習 → TYテスト で現行仕様をBT（リークフリー）。
本番ファイルは全てバックアップ→finallyで必ず復元。モデルは_bt別ファイルに出るので本番モデルは無傷。
使い方: python validate_multiyear.py 2024
"""
import os, sys, shutil, subprocess, time

TY = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
PY = sys.executable

# 退避対象（<=(TY-1)版で上書きされる本番アーティファクト）
PROD = ["race_features.csv", "course_bias.csv",
        "sire_stats_train.csv", "sire_stats_father_train.csv", "sire_stats_bms_train.csv",
        "model_mf_result.csv", "model_result.csv", "model_result_place2.csv", "model_result_place3.csv"]


def backup():
    for f in PROD:
        if os.path.exists(f):
            shutil.copy(f, f + ".prodbak")
    print(f"[{time.strftime('%H:%M')}] 本番{sum(os.path.exists(f+'.prodbak') for f in PROD)}ファイル退避完了", flush=True)


def restore():
    n = 0
    for f in PROD:
        b = f + ".prodbak"
        if os.path.exists(b):
            shutil.copy(b, f)
            os.remove(b)
            n += 1
    for f in ["model_mf_bt.pkl", "model_bt.pkl"]:
        if os.path.exists(f):
            os.remove(f)
    print(f"[{time.strftime('%H:%M')}] 本番{n}ファイル復元＋BTモデル削除完了", flush=True)


def step(msg, cmd, env=None):
    print(f"\n{'='*50}\n[{time.strftime('%H:%M')}] {msg}\n{'='*50}", flush=True)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"失敗: {msg} (rc={r.returncode})")


def main():
    t0 = time.time()
    env = dict(os.environ, KEIBA_TEST_YEAR=str(TY), PYTHONUTF8="1")
    try:
        backup()
        # 1. 血統統計を <=(TY-1) で再集計（_train上書き）
        print(f"\n[{time.strftime('%H:%M')}] 血統統計 <={TY-1} 再集計...", flush=True)
        from sire_stats import build_sire_stats
        build_sire_stats(max_year=TY - 1, suffix="_train")
        # 2. 特徴量を <=(TY-1) 集計で再生成（course_bias<=TY-1 ＋ sire_train<=TY-1）
        print(f"\n[{time.strftime('%H:%M')}] 特徴量再生成 year_max={TY-1}...", flush=True)
        from features import build_features
        build_features(year_max=TY - 1)
        # 3. MF・主モデルを <=(TY-1)学習 / TY検証
        step(f"MFモデル学習 (<={TY-1}/test{TY})", [PY, "train_mf_v2.py", "backtest"], env)
        step(f"主モデル学習 (<={TY-1}/test{TY})", [PY, "model.py", "backtest"], env)
        # 4. 現行仕様BT（{TY}実払戻）
        step(f"現行仕様BT test{TY}", [PY, "backtest_spec_2025.py"], env)
        print(f"\n[{time.strftime('%H:%M')}] 全工程完了 ({(time.time()-t0)/60:.0f}分)", flush=True)
    finally:
        restore()


if __name__ == "__main__":
    main()
