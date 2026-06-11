import pandas as pd
import numpy as np
import re


def clean_race_data(
    input_csv="race_data_large.csv",
    output_csv="race_data_clean.csv",
):
    df = pd.read_csv(input_csv, low_memory=False)

    print(f"読み込み完了: {len(df)}行")
    print(f"列一覧: {list(df.columns)}")

    base_columns = [
        "着順", "枠番", "馬番", "馬名", "性齢", "斤量", "騎手",
        "タイム", "着差", "タイム指数", "タイム指数M", "スタート指数",
        "追走指数", "上がり指数", "通過", "上り", "単勝オッズ", "人気",
        "馬体重_raw", "調教タイム", "厩舎コメント", "備考", "調教師", "馬主", "賞金",
    ]

    # 追加列（race情報 + 【新規】horse_id）
    extra_cols = []
    for col in ["距離", "馬場種別", "馬場状態", "レースクラス", "race_id", "horse_id"]:
        if col in df.columns:
            extra_cols.append(col)

    # ベース列だけ抽出して列名付与
    base_df = df.iloc[:, :len(base_columns)].copy()
    base_df.columns = base_columns[: base_df.shape[1]]

    # 追加列を結合
    for col in extra_cols:
        base_df[col] = df[col].values

    df = base_df.copy()

    # ** を除去
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.replace(r"^\*+$", "", regex=True)

    # 馬体重を分割
    df["馬体重"]  = df["馬体重_raw"].str.extract(r"(\d+)").astype(float)
    df["体重増減"] = df["馬体重_raw"].str.extract(r"\(([+-]?\d+)\)").astype(float)
    df = df.drop(columns=["馬体重_raw"])

    # 性別・年齢を分割
    df["性別"] = df["性齢"].str.extract(r"([牡牝セ])")
    df["年齢"] = df["性齢"].str.extract(r"(\d+)").astype(float)

    # 着順を数値に
    df["着順_num"] = pd.to_numeric(df["着順"], errors="coerce")

    # 数値変換
    for col in ["単勝オッズ", "人気", "斤量", "上り", "賞金"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 馬場状態を数値に
    if "馬場状態" in df.columns:
        baba_map = {"良": 1, "稍重": 2, "重": 3, "不良": 4}
        df["馬場状態_num"] = df["馬場状態"].map(baba_map)

    # 馬場種別を数値に
    if "馬場種別" in df.columns:
        df["is_turf"] = (df["馬場種別"] == "芝").astype(float)

    # レースクラスを数値に
    # netkeibaの表記ゆれ（新表記/旧表記/年齢条件付き等）に対応するため
    # 完全一致ではなく部分一致（キーワード検索）で判定する
    if "レースクラス" in df.columns:
        def _classify(s):
            if pd.isna(s):
                return np.nan
            s = str(s)
            # グレードレース（G1/G2/G3、JpnI等も含む）
            if "G1" in s or "GI" in s.replace("Ⅰ","I").replace("Ｉ","I") or "Jpn1" in s or "JpnI" in s:
                return 9
            if "G2" in s or "GII" in s.replace("Ⅱ","II") or "Jpn2" in s or "JpnII" in s:
                return 8
            if "G3" in s or "GIII" in s.replace("Ⅲ","III") or "Jpn3" in s or "JpnIII" in s:
                return 7
            # オープン・リステッド
            if "オープン" in s or "OP" in s or "L" == s.strip() or "リステッド" in s:
                return 6
            # 新表記（条件クラス）
            if "3勝クラス" in s or "1600万" in s:
                return 5
            if "2勝クラス" in s or "1000万" in s:
                return 4
            if "1勝クラス" in s or "500万" in s:
                return 3
            if "未勝利" in s:
                return 2
            if "新馬" in s:
                return 1
            return np.nan

        df["クラス_num"] = df["レースクラス"].apply(_classify)
        unmapped = df["クラス_num"].isna().sum()
        if unmapped > 0:
            print(f"  クラス_num: 未分類 {unmapped}行 "
                  f"(例: {df.loc[df['クラス_num'].isna(), 'レースクラス'].dropna().unique()[:5]})")

    # 不要列を除外
    drop_cols = [
        "タイム指数", "タイム指数M", "スタート指数",
        "追走指数", "上がり指数", "調教タイム",
        "厩舎コメント", "備考",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n保存完了 → {output_csv}")
    print(f"行数: {len(df)}, 列数: {len(df.columns)}")
    print(f"列一覧: {list(df.columns)}")
    return df


if __name__ == "__main__":
    df = clean_race_data(
        input_csv="race_data_large.csv",
        output_csv="race_data_clean.csv",
    )