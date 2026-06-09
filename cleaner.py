import pandas as pd
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
    if "レースクラス" in df.columns:
        class_map = {
            "新馬": 1, "未勝利": 2, "1勝クラス": 3,
            "2勝クラス": 4, "3勝クラス": 5,
            "オープン": 6, "G3": 7, "G2": 8, "G1": 9,
        }
        df["クラス_num"] = df["レースクラス"].map(class_map)

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