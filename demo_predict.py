"""
demo_predict.py
────────────────
keiba_predict.py のデモ実行用スクリプト。
平日でnetkeibaから出馬表が取得できないため、
race_features.csv の実レースデータを「出馬表」代わりに使い
予測〜印付け〜レポート表示までの流れを確認する。

使い方:
  python demo_predict.py            # ランダムな2025年レースで実行
  python demo_predict.py <race_id>  # race_idを指定して実行
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

BASE_DIR = r"c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai"
sys.path.insert(0, BASE_DIR)

from keiba_predict import (
    calc_place_probs_harvill, kelly_fraction, build_report, JYO_NAMES
)


def main():
    print("モデル読み込み中...")
    with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
        saved = pickle.load(f)
    models   = saved["models"]
    use_cols = saved["use_cols"]

    mf_models = mf_cols = None
    mf_path = os.path.join(BASE_DIR, "model_mf.pkl")
    if os.path.exists(mf_path):
        with open(mf_path, "rb") as f:
            mf_saved = pickle.load(f)
        mf_models = mf_saved["models"]
        mf_cols   = mf_saved["use_cols"]
        print("  市場フリーモデル読み込み完了")

    print("特徴量データ読み込み中...")
    df = pd.read_csv(os.path.join(BASE_DIR, "race_features.csv"), low_memory=False)
    df["年"] = df["race_id"].astype(str).str[:4].astype(int)

    if len(sys.argv) > 1:
        race_id = sys.argv[1]
        pdf = df[df["race_id"].astype(str) == str(race_id)].copy()
        if pdf.empty:
            print(f"race_id {race_id} のデータが見つかりません")
            sys.exit(1)
    else:
        # 2025年データから出走頭数10頭以上のレースをランダムに1つ選ぶ
        df_2025 = df[df["年"] == 2025]
        candidates = df_2025.groupby("race_id").size()
        candidates = candidates[candidates >= 10].index.tolist()
        race_id = str(np.random.choice(candidates))
        pdf = df_2025[df_2025["race_id"].astype(str) == str(race_id)].copy()

    print(f"\n=== デモ対象レース: {race_id} ===\n")

    race_id = str(race_id)
    jyo_cd  = int(race_id[4:6])
    race_no = int(race_id[10:12])
    jyo_name = JYO_NAMES.get(jyo_cd, f"競馬場{jyo_cd}")

    # ── 予測 ──
    X = pdf.reindex(columns=use_cols)
    preds = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    pdf["予測スコア"] = preds
    pdf["予測順位"]   = pdf["予測スコア"].rank(ascending=False).astype(int)

    raw = np.nan_to_num(preds, nan=0.0)
    raw = np.clip(raw, 0, None)
    win_probs = raw / raw.sum() if raw.sum() > 0 else np.ones(len(raw)) / len(raw)
    pdf["勝ち確率"] = win_probs

    place2, place3 = calc_place_probs_harvill(win_probs)
    pdf["連対確率"]  = place2
    pdf["複勝確率"]  = place3
    pdf["3着内確率"] = place3

    pdf["単勝期待値"] = pdf["勝ち確率"] * pdf["単勝オッズ"] - 1
    pdf["推奨賭け率"] = pdf.apply(
        lambda r: kelly_fraction(r["勝ち確率"], r["単勝オッズ"]), axis=1
    )

    # ── 市場フリーモデル ──
    pdf["MF予測順位"] = np.nan
    pdf["乖離スコア"] = np.nan
    pdf["MF勝ち確率"] = np.nan
    if mf_models is not None:
        X_mf = pdf.reindex(columns=mf_cols)
        mf_preds = np.mean([m.predict_proba(X_mf)[:, 1] for m in mf_models], axis=0)
        pdf["MF予測順位"] = pd.Series(mf_preds).rank(ascending=False).values
        pdf["乖離スコア"] = pdf["予測順位"] - pdf["MF予測順位"]
        mf_raw = np.clip(np.nan_to_num(mf_preds, nan=0.0), 0, None)
        pdf["MF勝ち確率"] = mf_raw / mf_raw.sum() if mf_raw.sum() > 0 else np.ones(len(mf_raw)) / len(mf_raw)

    # ── 戦略判定（keiba_predict.pyと同一ロジック） ──
    def check_strategy(row):
        s = []
        if row["予測順位"] == 1 and row["単勝期待値"] >= 0.3 and 1.5 <= row["単勝オッズ"] <= 20:
            s.append("戦略A")
            if row["人気"] != 1:
                s.append("戦略A-2")
        if row["人気"] >= 3 and row["予測順位"] == 1 and row["単勝期待値"] >= 0.3:
            s.append("戦略C")
        if pd.notna(row.get("前走間隔")) and 2 <= row["前走間隔"] <= 4 \
                and row["予測順位"] == 1 and row["単勝期待値"] >= 0.2:
            s.append("戦略D")
        jyo = int(str(row.get("race_id", "000000000000"))[4:6])
        if jyo in [5, 7] and row["予測順位"] == 1 and row["単勝期待値"] >= 0.3 \
                and 1.5 <= row["単勝オッズ"] <= 20:
            s.append("戦略F(東京・中京)")
        if jyo in [6, 10] and row["予測順位"] == 1 and row["単勝期待値"] >= 0.3 \
                and 1.5 <= row["単勝オッズ"] <= 20:
            s.append("戦略H(中山・小倉)")
        if jyo in [5, 7] and pd.notna(row.get("距離")) and row.get("距離", 9999) <= 1400 \
                and row["予測順位"] == 1 and row["単勝期待値"] >= 0.3 \
                and 1.5 <= row["単勝オッズ"] <= 20:
            s.append("戦略FG(東京・中京×短距離)")
        return " / ".join(s) if s else ""

    pdf["該当戦略"] = pdf.apply(check_strategy, axis=1)

    # ── レース概要 ──
    dist = int(pdf["距離"].iloc[0]) if pd.notna(pdf["距離"].iloc[0]) else "不明"
    turf = "芝" if pdf["is_turf"].iloc[0] == 1 else "ダート"
    baba_inv = {1: "良", 2: "稍重", 3: "重", 4: "不良"}
    baba = baba_inv.get(pdf["馬場状態_num"].iloc[0], "不明")
    cls_inv = {1:"新馬",2:"未勝利",3:"1勝クラス",4:"2勝クラス",
               5:"3勝クラス",6:"オープン",7:"G3",8:"G2",9:"G1"}
    cls = cls_inv.get(pdf["クラス_num"].iloc[0], "不明")
    n_horse = len(pdf)

    report = build_report(pdf, race_id, jyo_name, race_no, dist, turf, baba, cls, n_horse)
    print(report)

    # ── today_predictions.csv に保存（ダッシュボード確認用デモ） ──
    try:
        pdf["_ai_rank"] = pdf["勝ち確率"].rank(ascending=False)
        n = len(pdf)
        pdf["総合スコア"] = (
            (1 - (pdf["_ai_rank"] - 1) / n) * 80 +
            pdf["該当戦略"].apply(lambda s: 20 if s else 0)
        )
        final_top = pdf.sort_values("総合スコア", ascending=False).head(5)
        marks = ["◎", "○", "▲", "△", "×"]
        pdf["推奨ランク"] = ""
        for i, (idx, _) in enumerate(final_top.iterrows()):
            if i < len(marks):
                pdf.at[idx, "推奨ランク"] = marks[i]

        save_cols = [
            "race_id", "馬名", "馬番", "枠番",
            "単勝オッズ", "人気",
            "勝ち確率", "MF勝ち確率", "複勝確率", "3着内確率",
            "単勝期待値", "推奨賭け率",
            "乖離スコア", "MF予測順位",
            "該当戦略", "推奨ランク", "総合スコア",
            "予測順位", "過去勝率", "過去出走数", "前走間隔",
        ]
        save_cols = [c for c in save_cols if c in pdf.columns]
        save_df = pdf[save_cols].copy()

        save_df["jyo"]      = jyo_name
        save_df["race_no"]  = race_no
        save_df["距離"]     = dist
        save_df["馬場"]     = turf
        save_df["馬場状態"] = baba
        save_df["クラス"]   = cls
        save_df["予想日時"] = pd.Timestamp.now().strftime("%Y/%m/%d %H:%M") + "（デモ）"

        out_path = os.path.join(BASE_DIR, "today_predictions.csv")
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path)
            existing = existing[existing["race_id"].astype(str) != str(race_id)]
            save_df = pd.concat([existing, save_df], ignore_index=True)
        save_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n  デモ予想データ保存完了 → {out_path}")
    except Exception as e:
        print(f"  デモ保存エラー: {e}")
        import traceback; traceback.print_exc()

    # ── 答え合わせ（実際の着順） ──
    print("\n" + "=" * 40)
    print("📋 答え合わせ（実際の結果）")
    print("=" * 40)
    actual = pdf.sort_values("着順_num")[["着順_num", "馬名", "予測順位", "単勝オッズ", "該当戦略"]].head(5)
    for _, row in actual.iterrows():
        mark = "✅" if row["予測順位"] == 1 and row["着順_num"] == 1 else ""
        print(f"  {int(row['着順_num'])}着  {row['馬名']:12} (予測{int(row['予測順位'])}位 / "
              f"オッズ{row['単勝オッズ']}) {row['該当戦略']} {mark}")


if __name__ == "__main__":
    main()