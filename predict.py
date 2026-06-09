import pandas as pd
import numpy as np
import lightgbm as lgb
import requests
from bs4 import BeautifulSoup
import time
import re
import pickle

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
}

FEATURE_COLS = [
    "枠番", "馬番", "斤量", "斤量_相対",
    "年齢", "is_male", "is_female", "is_castrated",
    "馬体重", "体重増減", "馬体重_相対",
    "人気", "上り", "出走頭数", "競馬場cd", "レース番号",
    "過去出走数", "過去平均着順", "過去勝率", "過去複勝率",
    "過去平均上り", "直近3走平均着順",
    "過去平均タイム秒", "直近3走平均タイム秒", "過去最速タイム秒",
    "直近3走平均上り", "過去平均体重増減",
    "距離カテゴリ", "距離別過去平均着順",
    "騎手勝率", "騎手複勝率", "調教師勝率", "調教師複勝率",
    "距離", "馬場状態_num", "is_turf", "クラス_num",
]


def save_model(models, path="model.pkl"):
    """学習済みモデルを保存"""
    import pickle
    with open(path, "wb") as f:
        pickle.dump(models, f)
    print(f"モデル保存 → {path}")


def load_model(path="model.pkl"):
    """保存済みモデルを読み込み"""
    import pickle
    with open(path, "rb") as f:
        models = pickle.load(f)
    print(f"モデル読み込み → {path}")
    return models


def get_upcoming_race(race_id):
    """当日・翌日のレースデータを取得（オッズ・出走表）"""
    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        response = requests.get(url, headers=HEADERS)
        response.encoding = "EUC-JP"
        soup = BeautifulSoup(response.text, "html.parser")

        race_info = {}
        race_data_div = soup.find("div", class_="data_intro")
        if race_data_div:
            text = race_data_div.get_text()
            dist_match = re.search(r'(\d{3,4})m', text)
            race_info["距離"] = int(dist_match.group(1)) if dist_match else None
            race_info["馬場種別"] = "芝" if "芝" in text else "ダート"
            race_info["馬場状態"] = None
            for condition in ["不良", "重", "稍重", "良"]:
                if condition in text:
                    race_info["馬場状態"] = condition
                    break
            race_info["レースクラス"] = None
            for cls in ["新馬", "未勝利", "1勝クラス", "2勝クラス",
                       "3勝クラス", "オープン", "G1", "G2", "G3"]:
                if cls in text:
                    race_info["レースクラス"] = cls
                    break

        table = soup.find("table", class_="race_table_01")
        if table is None:
            print(f"出走表が見つかりません: {race_id}")
            return None

        headers = [th.get_text(strip=True) for th in table.find_all("tr")[0].find_all("th")]
        rows = []
        for tr in table.find_all("tr")[1:]:
            cols = [td.get_text(strip=True) for td in tr.find_all("td")]
            if cols:
                rows.append(cols)

        df = pd.DataFrame(rows)
        if headers and len(headers) == df.shape[1]:
            df.columns = headers
        for key, val in race_info.items():
            df[key] = val
        df["race_id"] = str(race_id)
        return df

    except Exception as e:
        print(f"エラー: {race_id} → {e}")
        return None


def build_predict_features(race_df, history_df):
    """
    取得したレースデータに過去成績を付与して予測用特徴量を作る
    history_df: race_features.csvを読み込んだもの
    """
    # 馬名ごとの過去成績を集計
    horse_stats = {}
    for horse, group in history_df.groupby("馬名"):
        valid = group.dropna(subset=["着順_num"])
        if len(valid) == 0:
            continue
        last3 = valid.tail(3)
        horse_stats[horse] = {
            "過去出走数":           len(valid),
            "過去平均着順":         valid["着順_num"].mean(),
            "過去勝率":             (valid["着順_num"] == 1).mean(),
            "過去複勝率":           (valid["着順_num"] <= 3).mean(),
            "過去平均上り":         valid["上り"].mean(),
            "直近3走平均着順":     last3["着順_num"].mean(),
            "過去平均タイム秒":     valid["タイム秒"].mean() if "タイム秒" in valid.columns else np.nan,
            "直近3走平均タイム秒": last3["タイム秒"].mean() if "タイム秒" in last3.columns else np.nan,
            "過去最速タイム秒":     valid["タイム秒"].min() if "タイム秒" in valid.columns else np.nan,
            "直近3走平均上り":     last3["上り"].mean(),
            "過去平均体重増減":     valid["体重増減"].mean(),
        }

    # 騎手・調教師の勝率
    jockey_stats = {}
    trainer_stats = {}
    for _, row in history_df.iterrows():
        j = row.get("騎手", "")
        t = row.get("調教師", "")
        chakujun = row.get("着順_num", np.nan)
        if pd.isna(chakujun):
            continue
        for stats, key in [(jockey_stats, j), (trainer_stats, t)]:
            if key not in stats:
                stats[key] = {"runs": 0, "wins": 0, "top3": 0}
            stats[key]["runs"] += 1
            if chakujun == 1:
                stats[key]["wins"] += 1
            if chakujun <= 3:
                stats[key]["top3"] += 1

    # 予測用データに特徴量を付与
    rows = []
    for _, row in race_df.iterrows():
        horse = row.get("馬名", "")
        jockey = row.get("騎手", "")
        trainer = row.get("調教師", "")

        feat = {
            "馬名":    horse,
            "race_id": row.get("race_id", ""),
            "単勝オッズ": pd.to_numeric(row.get("単勝オッズ", np.nan), errors="coerce"),
            "人気":    pd.to_numeric(row.get("人気", np.nan), errors="coerce"),
            "枠番":    pd.to_numeric(row.get("枠番", np.nan), errors="coerce"),
            "馬番":    pd.to_numeric(row.get("馬番", np.nan), errors="coerce"),
            "斤量":    pd.to_numeric(row.get("斤量", np.nan), errors="coerce"),
            "距離":    pd.to_numeric(row.get("距離", np.nan), errors="coerce"),
            "is_turf": 1 if row.get("馬場種別") == "芝" else 0,
        }

        # 馬場状態
        baba_map = {"良": 1, "稍重": 2, "重": 3, "不良": 4}
        feat["馬場状態_num"] = baba_map.get(row.get("馬場状態"), np.nan)

        # クラス
        class_map = {"新馬": 1, "未勝利": 2, "1勝クラス": 3, "2勝クラス": 4,
                     "3勝クラス": 5, "オープン": 6, "G3": 7, "G2": 8, "G1": 9}
        feat["クラス_num"] = class_map.get(row.get("レースクラス"), np.nan)

        # 性別・年齢
        seire = str(row.get("性齢", ""))
        feat["is_male"]      = 1 if "牡" in seire else 0
        feat["is_female"]    = 1 if "牝" in seire else 0
        feat["is_castrated"] = 1 if "セ" in seire else 0
        age_match = re.search(r'(\d+)', seire)
        feat["年齢"] = int(age_match.group(1)) if age_match else np.nan

        # 馬体重
        weight_raw = str(row.get("馬体重", ""))
        w_match = re.search(r'(\d+)', weight_raw)
        feat["馬体重"] = int(w_match.group(1)) if w_match else np.nan
        d_match = re.search(r'\(([+-]?\d+)\)', weight_raw)
        feat["体重増減"] = int(d_match.group(1)) if d_match else np.nan

        # 過去成績
        hs = horse_stats.get(horse, {})
        for key in ["過去出走数", "過去平均着順", "過去勝率", "過去複勝率",
                    "過去平均上り", "直近3走平均着順", "過去平均タイム秒",
                    "直近3走平均タイム秒", "過去最速タイム秒",
                    "直近3走平均上り", "過去平均体重増減"]:
            feat[key] = hs.get(key, np.nan)

        # 騎手・調教師
        js = jockey_stats.get(jockey, {})
        feat["騎手勝率"]   = js["wins"] / js["runs"] if js.get("runs", 0) > 0 else np.nan
        feat["騎手複勝率"] = js["top3"] / js["runs"] if js.get("runs", 0) > 0 else np.nan
        ts = trainer_stats.get(trainer, {})
        feat["調教師勝率"]   = ts["wins"] / ts["runs"] if ts.get("runs", 0) > 0 else np.nan
        feat["調教師複勝率"] = ts["top3"] / ts["runs"] if ts.get("runs", 0) > 0 else np.nan

        rows.append(feat)

    predict_df = pd.DataFrame(rows)

    # レース内の相対特徴量
    predict_df["出走頭数"]    = len(predict_df)
    predict_df["馬体重_相対"] = predict_df["馬体重"] - predict_df["馬体重"].mean()
    predict_df["斤量_相対"]   = predict_df["斤量"]   - predict_df["斤量"].mean()
    predict_df["人気順位"]    = predict_df["人気"].rank()
    predict_df["人気_inv"]    = 1 / predict_df["人気"].clip(lower=1)

    # race_idから情報を分解
    race_id_str = str(predict_df["race_id"].iloc[0])
    predict_df["競馬場cd"] = int(race_id_str[4:6])
    predict_df["レース番号"] = int(race_id_str[10:12])
    predict_df["距離カテゴリ"] = pd.cut(
        predict_df["レース番号"],
        bins=[0, 4, 8, 12], labels=[1, 2, 3]
    ).astype(float)
    predict_df["距離別過去平均着順"] = np.nan
    predict_df["上り"] = np.nan  # 当日は未知

    return predict_df


def predict_race(race_id, model_path="model.pkl",
                 history_csv="race_features.csv"):
    """レースIDを指定して予測結果を表示"""

    print(f"\nレース {race_id} の予測を開始...")

    # モデル読み込み
    models = load_model(model_path)

    # 出走表取得
    race_df = get_upcoming_race(race_id)
    if race_df is None:
        print("出走表の取得に失敗しました")
        return

    # 過去成績データ読み込み
    print("過去成績データ読み込み中...")
    history_df = pd.read_csv(history_csv)

    # 特徴量構築
    predict_df = build_predict_features(race_df, history_df)

    # 予測
    X = predict_df[[c for c in FEATURE_COLS if c in predict_df.columns]]
    preds = np.mean([m.predict(X) for m in models], axis=0)
    predict_df["予測着順スコア"] = preds
    predict_df["予測順位"] = predict_df["予測着順スコア"].rank()

    # 結果表示
    result = predict_df[["馬名", "予測順位", "予測着順スコア",
                          "単勝オッズ", "人気"]].sort_values("予測順位")

    print(f"\n{'='*50}")
    print(f"レース {race_id} 予測結果")
    print(f"{'='*50}")
    print(result.to_string(index=False))

    # 戦略4（中穴）に該当する馬を強調
    buy = result[
        (result["予測順位"] == 1) &
        (result["単勝オッズ"] >= 3) &
        (result["単勝オッズ"] <= 10)
    ]
    if len(buy) > 0:
        print(f"\n🎯 戦略4（中穴）推奨馬:")
        print(buy[["馬名", "単勝オッズ", "人気"]].to_string(index=False))
    else:
        print(f"\n予測1位: {result.iloc[0]['馬名']}（オッズ {result.iloc[0]['単勝オッズ']}倍）")

    return result


if __name__ == "__main__":
    # 使い方：race_idを指定して実行
    # 例: 今週の東京1Rを予測したい場合
    # race_id = "202405050101"  # 2024年・東京・5回・1日目・1R

    race_id = input("予測するレースIDを入力してください（例: 202405050101）: ")
    predict_race(race_id)