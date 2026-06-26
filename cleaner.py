# -*- coding: utf-8 -*-
"""
features前処理 修正版（clean_race_data）
─────────────────────────────────────────
修正点:
  1. 【最重要】追加列(horse_id/賞金/距離/クラス_num/回り 等)を
     位置スライスでなく「列名」で取得し、列順の変化による
     中身ズレを防止する。
  2. クラス_num の二重生成を解消。
     scraper側で生成済みの クラス_num を優先採用し、
     欠損している行だけ前処理側のキーワード判定で補完する。
     （番号体系も scraper 側 1-8 に統一）

設計方針:
  ・テーブル由来のベース列（着順〜馬主）は、ヘッダー名が信用
    できないため従来どおり「位置」で取得する。
  ・スクレイパーが付与した追加列は「列名」で取得する（=順番非依存）。
  → 両者のいいとこ取りで事故を防ぐ。
"""
import pandas as pd
import numpy as np
import re


# scraper側と統一したクラス序列（1-8）
CLASS_MAP = {
    "新馬・未勝利": 1, "1勝クラス": 2, "2勝クラス": 3, "3勝クラス": 4,
    "オープン特別・L": 5, "G3": 6, "G2": 7, "G1": 8,
}


def classify_class(s):
    """レースクラス文字列 → クラス_num(1-8)。scraper側が欠損のときの保険。"""
    if pd.isna(s):
        return np.nan
    s = str(s)
    su = s.replace("Ⅰ", "I").replace("Ⅱ", "II").replace("Ⅲ", "III")
    # グレード
    if "G1" in su or "GI" in su and "GII" not in su and "GIII" not in su or "Jpn1" in su or "JpnI" in su and "JpnII" not in su:
        return 8
    if "G2" in su or "GII" in su and "GIII" not in su or "Jpn2" in su:
        return 7
    if "G3" in su or "GIII" in su or "Jpn3" in su:
        return 6
    if "オープン" in s or s.strip() in ("L", "OP") or "リステッド" in s:
        return 5
    if "3勝" in s or "1600万" in s:
        return 4
    if "2勝" in s or "1000万" in s:
        return 3
    if "1勝" in s or "500万" in s:
        return 2
    if "未勝利" in s or "新馬" in s:
        return 1
    return np.nan


def clean_race_data(input_csv="race_data_large.csv",
                    output_csv="race_data_clean.csv"):
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"読み込み完了: {len(df)}行")
    print(f"列一覧: {list(df.columns)}")

    # ── テーブル由来のベース列（位置で取得）──────────────────
    # ※ スクレイパーが結果テーブルから作る列の並び。先頭から順に対応。
    base_columns = [
        "着順", "枠番", "馬番", "馬名", "性齢", "斤量", "騎手",
        "タイム", "着差", "タイム指数", "タイム指数M", "スタート指数",
        "追走指数", "上がり指数", "通過", "上り", "単勝オッズ", "人気",
        "馬体重_raw", "調教タイム", "厩舎コメント", "備考", "調教師", "馬主",
    ]
    # 追加列(horse_id/賞金)が始まる前までを位置で切る
    base_df = df.iloc[:, :len(base_columns)].copy()
    base_df.columns = base_columns[:base_df.shape[1]]

    # ── 追加列（列名で取得・順番非依存）──────────────────────
    name_based = ["距離", "馬場種別", "馬場状態", "回り",
                  "レースクラス", "クラス_num", "レース名",
                  "賞金", "horse_id", "race_id"]
    for col in name_based:
        if col in df.columns:
            base_df[col] = df[col].values
        else:
            print(f"  ⚠ 列なし（旧データ?）: {col}")

    df = base_df

    # ── ** 除去 ──────────────────────────────────────────
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.replace(r"^\*+$", "", regex=True)

    # ── 馬体重分割 ───────────────────────────────────────
    if "馬体重_raw" in df.columns:
        df["馬体重"] = df["馬体重_raw"].astype(str).str.extract(r"(\d+)").astype(float)
        df["体重増減"] = df["馬体重_raw"].astype(str).str.extract(r"\(([+-]?\d+)\)").astype(float)
        df = df.drop(columns=["馬体重_raw"])

    # ── 性別・年齢 ───────────────────────────────────────
    df["性別"] = df["性齢"].astype(str).str.extract(r"([牡牝セ])")
    df["年齢"] = df["性齢"].astype(str).str.extract(r"(\d+)").astype(float)

    # ── 数値化 ───────────────────────────────────────────
    df["着順_num"] = pd.to_numeric(df["着順"], errors="coerce")
    for col in ["単勝オッズ", "人気", "斤量", "上り", "賞金"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 馬場状態・馬場種別 ────────────────────────────────
    if "馬場状態" in df.columns:
        df["馬場状態_num"] = df["馬場状態"].map({"良": 1, "稍重": 2, "重": 3, "不良": 4})
    if "馬場種別" in df.columns:
        df["is_turf"] = (df["馬場種別"] == "芝").astype(float)

    # ── クラス_num：scraper優先・欠損だけ補完（二重生成の解消）──
    if "クラス_num" in df.columns:
        df["クラス_num"] = pd.to_numeric(df["クラス_num"], errors="coerce")
        before_na = df["クラス_num"].isna().sum()
        if before_na > 0 and "レースクラス" in df.columns:
            fill = df.loc[df["クラス_num"].isna(), "レースクラス"].apply(classify_class)
            df.loc[df["クラス_num"].isna(), "クラス_num"] = fill
        after_na = df["クラス_num"].isna().sum()
        print(f"  クラス_num: scraper欠損 {before_na}行 → 補完後欠損 {after_na}行")
        if after_na > 0:
            ex = df.loc[df["クラス_num"].isna(), "レースクラス"].dropna().unique()[:5]
            print(f"    未分類サンプル: {ex}")
    elif "レースクラス" in df.columns:
        # 旧データのみのケース（全部前処理側で生成）
        df["クラス_num"] = df["レースクラス"].apply(classify_class)

    # ── 回りを数値化（任意・特徴量で使いやすく）────────────
    if "回り" in df.columns:
        df["回り_num"] = df["回り"].map({"右": 1, "左": 2, "直": 3})

    # ── 不要列除外 ───────────────────────────────────────
    drop_cols = ["タイム指数", "タイム指数M", "スタート指数",
                 "追走指数", "上がり指数", "調教タイム", "厩舎コメント", "備考"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n保存完了 → {output_csv}")
    print(f"行数: {len(df)}, 列数: {len(df.columns)}")
    print(f"列一覧: {list(df.columns)}")

    # ── 欠損サマリ（再取得が効いたかの確認用）──────────────
    print("\n主要列の欠損率:")
    for col in ["horse_id", "クラス_num", "賞金", "回り"]:
        if col in df.columns:
            r = df[col].isna().mean() * 100 if df[col].dtype != object \
                else (df[col].astype(str).str.strip().isin(["", "nan", "None"]).mean() * 100)
            print(f"  {col}: {r:.2f}%")
    return df


if __name__ == "__main__":
    clean_race_data(input_csv="race_data_large.csv",
                    output_csv="race_data_clean.csv")