import pandas as pd
import numpy as np
from datetime import datetime


def load_and_prepare(csv_path="race_data_clean.csv"):
    df = pd.read_csv(csv_path, low_memory=False)

    def time_to_sec(t):
        try:
            if pd.isna(t) or t == "":
                return np.nan
            parts = str(t).split(":")
            return int(parts[0]) * 60 + float(parts[1])
        except:
            return np.nan

    df["タイム秒"] = df["タイム"].apply(time_to_sec)

    df["race_id"]   = df["race_id"].astype(str)
    df["年"]        = df["race_id"].str[0:4].astype(int)
    df["競馬場cd"]  = df["race_id"].str[4:6].astype(int)
    df["回"]        = df["race_id"].str[6:8].astype(int)
    df["日"]        = df["race_id"].str[8:10].astype(int)
    df["レース番号"] = df["race_id"].str[10:12].astype(int)

    df["is_male"]      = (df["性別"] == "牡").astype(int)
    df["is_female"]    = (df["性別"] == "牝").astype(int)
    df["is_castrated"] = (df["性別"] == "セ").astype(int)
    df["人気_inv"]     = 1 / df["人気"].clip(lower=1)

    return df


def add_horse_history_features(df):
    """各馬の過去成績を特徴量として追加（当日データ混入なし・高速化版）"""

    df = df.sort_values(["馬名", "race_id"]).reset_index(drop=True)

    # ── 高速化：よく使う集計をgroupby一括処理（shift(1)でリーク防止） ──
    g = df.groupby("馬名")
    df["_win"]   = (df["着順_num"] == 1).astype(float)
    df["_top3"]  = (df["着順_num"] <= 3).astype(float)

    # 過去平均着順・勝率・複勝率（expanding + shift）
    df["過去出走数"]       = g["着順_num"].transform(lambda x: x.shift(1).expanding().count())
    df["過去平均着順"]     = g["着順_num"].transform(lambda x: x.shift(1).expanding().mean())
    df["過去勝率"]         = g["_win"].transform(lambda x: x.shift(1).expanding().mean())
    df["過去複勝率"]       = g["_top3"].transform(lambda x: x.shift(1).expanding().mean())
    df["直近3走平均着順"]  = g["着順_num"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["過去平均上り"]     = g["上り"].transform(lambda x: x.shift(1).expanding().mean())
    df["直近3走平均上り"]  = g["上り"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["過去最速上り"]     = g["上り"].transform(lambda x: x.shift(1).expanding().min())
    df["上り偏差"]         = g["上り"].transform(lambda x: x.shift(1).expanding().std())
    df["過去平均体重増減"] = g["体重増減"].transform(lambda x: x.shift(1).expanding().mean())

    # 前走情報
    df["前走着順"]   = g["着順_num"].transform(lambda x: x.shift(1))
    df["前走上り"]   = g["上り"].transform(lambda x: x.shift(1))

    # タイム系
    if "タイム秒" in df.columns:
        df["過去平均タイム秒"]      = g["タイム秒"].transform(lambda x: x.shift(1).expanding().mean())
        df["直近3走平均タイム秒"]   = g["タイム秒"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        df["過去最速タイム秒"]      = g["タイム秒"].transform(lambda x: x.shift(1).expanding().min())
    else:
        for col in ["過去平均タイム秒", "直近3走平均タイム秒", "過去最速タイム秒"]:
            df[col] = np.nan

    # 連続好走フラグ
    df["連続複勝フラグ"] = g["_top3"].transform(
        lambda x: x.shift(1).rolling(2, min_periods=2).min()
    )
    df["連続勝利フラグ"] = g["_win"].transform(
        lambda x: x.shift(1).rolling(2, min_periods=2).min()
    )

    # 近走改善度
    df["_last3_avg"] = g["着順_num"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["_all_avg"]   = g["着順_num"].transform(lambda x: x.shift(1).expanding().mean())
    df["近走改善度"] = df["_all_avg"] - df["_last3_avg"]

    # 距離変化
    df["前走距離"] = g["距離"].transform(lambda x: pd.to_numeric(x, errors="coerce").shift(1))
    df["距離変化"] = pd.to_numeric(df["距離"], errors="coerce") - df["前走距離"]

    df = df.drop(columns=["_win", "_top3", "_last3_avg", "_all_avg"], errors="ignore")

    result_rows = []

    for horse, group in df.groupby("馬名"):
        group = group.copy().reset_index(drop=True)

        for i in range(len(group)):
            row  = group.iloc[i].copy()
            past = group.iloc[:i]
            valid = past.dropna(subset=["着順_num"])

            if len(valid) == 0:
                row["過去出走数"]           = 0
                row["過去平均着順"]         = np.nan
                row["過去勝率"]             = np.nan
                row["過去複勝率"]           = np.nan
                row["過去平均上り"]         = np.nan
                row["直近3走平均着順"]      = np.nan
                row["過去平均タイム秒"]     = np.nan
                row["直近3走平均タイム秒"]  = np.nan
                row["過去最速タイム秒"]     = np.nan
                row["直近3走平均上り"]      = np.nan
                row["過去平均体重増減"]     = np.nan
                row["前走間隔"]             = np.nan
                row["同距離過去勝率"]       = np.nan
                row["同距離過去平均着順"]   = np.nan
                row["良馬場勝率"]           = np.nan
                row["重馬場勝率"]           = np.nan
                # 【新規】上がり関連
                row["過去最速上り"]         = np.nan
                row["上り偏差"]             = np.nan
                row["距離別過去平均上り"]   = np.nan
                row["前走着順"]             = np.nan
                row["前走上り"]             = np.nan
                row["前走距離"]             = np.nan
                row["距離変化"]             = np.nan
                row["連続複勝フラグ"]       = np.nan
                row["連続勝利フラグ"]       = np.nan
                row["近走改善度"]           = np.nan
                row["平均タイム差"]         = np.nan
            else:
                last3 = valid.tail(3)
                row["過去出走数"]           = len(valid)
                row["過去平均着順"]         = valid["着順_num"].mean()
                row["過去勝率"]             = (valid["着順_num"] == 1).mean()
                row["過去複勝率"]           = (valid["着順_num"] <= 3).mean()
                row["過去平均上り"]         = valid["上り"].mean()
                row["直近3走平均着順"]      = last3["着順_num"].mean()
                row["過去平均タイム秒"]     = valid["タイム秒"].mean() if "タイム秒" in valid.columns else np.nan
                row["直近3走平均タイム秒"]  = last3["タイム秒"].mean() if "タイム秒" in last3.columns else np.nan
                row["過去最速タイム秒"]     = valid["タイム秒"].min()  if "タイム秒" in valid.columns else np.nan
                row["直近3走平均上り"]      = last3["上り"].mean()
                row["過去平均体重増減"]     = valid["体重増減"].mean()

                # 【新規】上がり関連（過去データのみ・当日混入なし）
                row["過去最速上り"]   = valid["上り"].min()   # 最高（最小）上がり
                row["上り偏差"]       = valid["上り"].std()   # 上がりのばらつき

                # 前走間隔
                if len(past) > 0:
                    try:
                        d1 = datetime.strptime(str(row["race_id"])[:8], "%Y%m%d")
                        d2 = datetime.strptime(str(past.iloc[-1]["race_id"])[:8], "%Y%m%d")
                        row["前走間隔"] = (d1 - d2).days / 7
                    except:
                        row["前走間隔"] = np.nan
                else:
                    row["前走間隔"] = np.nan

                # 同距離過去成績（±200m）
                if "距離" in valid.columns:
                    current_dist = row.get("距離", np.nan)
                    if pd.notna(current_dist):
                        same_dist = valid[
                            (valid["距離"] >= current_dist - 200) &
                            (valid["距離"] <= current_dist + 200)
                        ]
                        if len(same_dist) > 0:
                            row["同距離過去勝率"]     = (same_dist["着順_num"] == 1).mean()
                            row["同距離過去平均着順"] = same_dist["着順_num"].mean()
                        else:
                            row["同距離過去勝率"]     = np.nan
                            row["同距離過去平均着順"] = np.nan
                    else:
                        row["同距離過去勝率"]     = np.nan
                        row["同距離過去平均着順"] = np.nan
                else:
                    row["同距離過去勝率"]     = np.nan
                    row["同距離過去平均着順"] = np.nan

                # 馬場状態別成績
                if "馬場状態" in valid.columns:
                    ryou = valid[valid["馬場状態"] == "良"]
                    omoi = valid[valid["馬場状態"].isin(["重", "不良"])]
                    row["良馬場勝率"] = (ryou["着順_num"] == 1).mean() if len(ryou) > 0 else np.nan
                    row["重馬場勝率"] = (omoi["着順_num"] == 1).mean() if len(omoi) > 0 else np.nan
                else:
                    row["良馬場勝率"] = np.nan
                    row["重馬場勝率"] = np.nan

                # 【新規】距離カテゴリ別過去平均上り
                if "レース番号" in valid.columns:
                    current_race_no = row.get("レース番号", np.nan)
                    if pd.notna(current_race_no):
                        if current_race_no <= 4:
                            dist_cat_filter = valid["レース番号"] <= 4
                        elif current_race_no <= 8:
                            dist_cat_filter = (valid["レース番号"] >= 5) & (valid["レース番号"] <= 8)
                        else:
                            dist_cat_filter = valid["レース番号"] >= 9
                        same_cat = valid[dist_cat_filter]
                        row["距離別過去平均上り"] = same_cat["上り"].mean() if len(same_cat) > 0 else np.nan
                    else:
                        row["距離別過去平均上り"] = np.nan
                else:
                    row["距離別過去平均上り"] = np.nan

                # ── 前走情報 ──────────────────────────────────────────
                if len(valid) > 0:
                    prev = valid.iloc[-1]
                    row["前走着順"]  = prev["着順_num"]
                    row["前走上り"]  = prev["上り"] if "上り" in prev.index else np.nan
                    row["前走距離"]  = pd.to_numeric(prev.get("距離", np.nan), errors="coerce")
                    # 距離変化（今回 - 前走）
                    cur_dist = pd.to_numeric(row.get("距離", np.nan), errors="coerce")
                    prev_dist = row["前走距離"]
                    row["距離変化"] = (cur_dist - prev_dist) if pd.notna(cur_dist) and pd.notna(prev_dist) else np.nan
                else:
                    row["前走着順"]  = np.nan
                    row["前走上り"]  = np.nan
                    row["前走距離"]  = np.nan
                    row["距離変化"]  = np.nan

                # ── 連続好走フラグ ─────────────────────────────────────
                if len(valid) >= 2:
                    last2 = valid.tail(2)
                    row["連続複勝フラグ"] = 1.0 if (last2["着順_num"] <= 3).all() else 0.0
                    row["連続勝利フラグ"] = 1.0 if (last2["着順_num"] == 1).all() else 0.0
                else:
                    row["連続複勝フラグ"] = np.nan
                    row["連続勝利フラグ"] = np.nan

                # ── 近走改善度（直近3走 vs 全過去の着順差） ────────────
                if len(valid) >= 4:
                    last3_avg = valid.tail(3)["着順_num"].mean()
                    all_avg   = valid["着順_num"].mean()
                    row["近走改善度"] = all_avg - last3_avg  # プラスほど改善中
                else:
                    row["近走改善度"] = np.nan

                # ── 勝ち馬とのタイム差平均 ─────────────────────────────
                if "タイム秒" in valid.columns and "着差" in valid.columns:
                    # 着差を秒に変換（例: "0.3" → 0.3秒）
                    def parse_chakusa(x):
                        try:
                            if pd.isna(x) or str(x).strip() in ["", "同着", "大差"]:
                                return np.nan
                            return float(str(x).replace(".", "."))
                        except:
                            return np.nan
                    chakusa_vals = valid["着差"].apply(parse_chakusa).dropna()
                    row["平均タイム差"] = chakusa_vals.mean() if len(chakusa_vals) > 0 else np.nan
                else:
                    row["平均タイム差"] = np.nan

            result_rows.append(row)

    return pd.DataFrame(result_rows).reset_index(drop=True)


def add_race_relative_features(df):
    """同じレース内での相対的な特徴量を追加"""
    rel_rows = []
    for race_id, group in df.groupby("race_id"):
        group = group.copy()
        group["出走頭数"]   = len(group)
        avg_weight          = group["馬体重"].mean()
        group["馬体重_相対"] = group["馬体重"] - avg_weight
        group["人気順位"]   = group["人気"].rank()
        group["斤量_相対"]  = group["斤量"] - group["斤量"].mean()
        rel_rows.append(group)
    return pd.concat(rel_rows, ignore_index=True)


def add_jockey_trainer_features(df):
    """騎手・調教師の累積勝率を特徴量として追加（リーク防止：その時点までのデータのみ）"""
    df = df.sort_values(["race_id"]).reset_index(drop=True)

    jockey_stats  = {}
    trainer_stats = {}
    jockey_winrate  = []
    jockey_fukusho  = []
    trainer_winrate = []
    trainer_fukusho = []

    for _, row in df.iterrows():
        jockey   = row["騎手"]
        trainer  = row["調教師"]
        chakujun = row["着順_num"]

        if jockey not in jockey_stats:
            jockey_stats[jockey] = {"runs": 0, "wins": 0, "top3": 0}
        if trainer not in trainer_stats:
            trainer_stats[trainer] = {"runs": 0, "wins": 0, "top3": 0}

        js = jockey_stats[jockey]
        ts = trainer_stats[trainer]
        jockey_winrate.append(js["wins"]  / js["runs"] if js["runs"] > 0 else np.nan)
        jockey_fukusho.append(js["top3"]  / js["runs"] if js["runs"] > 0 else np.nan)
        trainer_winrate.append(ts["wins"] / ts["runs"] if ts["runs"] > 0 else np.nan)
        trainer_fukusho.append(ts["top3"] / ts["runs"] if ts["runs"] > 0 else np.nan)

        if not np.isnan(chakujun):
            jockey_stats[jockey]["runs"]   += 1
            trainer_stats[trainer]["runs"] += 1
            if chakujun == 1:
                jockey_stats[jockey]["wins"]   += 1
                trainer_stats[trainer]["wins"] += 1
            if chakujun <= 3:
                jockey_stats[jockey]["top3"]   += 1
                trainer_stats[trainer]["top3"] += 1

    df["騎手勝率"]   = jockey_winrate
    df["騎手複勝率"] = jockey_fukusho
    df["調教師勝率"]   = trainer_winrate
    df["調教師複勝率"] = trainer_fukusho

    # ── 騎手×競馬場 勝率（その時点までのデータのみ） ──────────────────
    jockey_jyo_stats = {}   # {(騎手, 競馬場cd): {runs, wins}}
    jockey_jyo_winrate = []

    df2 = df.sort_values("race_id").reset_index(drop=True)
    for _, row in df2.iterrows():
        jockey  = row["騎手"]
        jyo_cd  = str(row["race_id"])[4:6]
        chakujun = row["着順_num"]
        key = (jockey, jyo_cd)

        if key not in jockey_jyo_stats:
            jockey_jyo_stats[key] = {"runs": 0, "wins": 0}
        st = jockey_jyo_stats[key]
        jockey_jyo_winrate.append(st["wins"] / st["runs"] if st["runs"] > 0 else np.nan)

        if not np.isnan(chakujun):
            st["runs"] += 1
            if chakujun == 1:
                st["wins"] += 1

    df2["騎手競馬場勝率"] = jockey_jyo_winrate
    # 元のdfに結合
    df = df.merge(df2[["race_id", "馬名", "騎手競馬場勝率"]],
                  on=["race_id", "馬名"], how="left")

    return df


def add_extra_advanced_features(df):
    """
    精度向上のための追加特徴量（高速化版）：
    ① 競馬場×距離 過去成績（groupby一括処理）
    ② 脚質（先行/差し/追込）
    ③ 開催時期（月・季節）
    """
    df = df.sort_values(["馬名", "race_id"]).reset_index(drop=True)

    # ── ① 脚質を通過順位から計算 ──────────────────────────────────────
    def calc_running_style(passage):
        try:
            if pd.isna(passage) or str(passage).strip() == "":
                return np.nan
            positions = [int(x) for x in str(passage).split("-") if x.isdigit()]
            return float(positions[0]) if positions else np.nan
        except:
            return np.nan

    if "通過" in df.columns:
        df["先行指数"] = df["通過"].apply(calc_running_style)
        df["過去平均先行指数"] = df.groupby("馬名")["先行指数"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        df["先行馬フラグ"] = (
            df["過去平均先行指数"] <= df["出走頭数"] / 3
        ).astype(float)
    else:
        df["過去平均先行指数"] = np.nan
        df["先行馬フラグ"]     = np.nan

    # ── ② 競馬場×距離 過去成績（高速化：groupby一括処理） ────────────
    if "競馬場cd" not in df.columns:
        df["競馬場cd"] = df["race_id"].astype(str).str[4:6].astype(int)
    df["距離_num"] = pd.to_numeric(df["距離"], errors="coerce")

    # 距離カテゴリ（±200m をカテゴリで近似）
    df["距離カテゴリ_jyo"] = (df["距離_num"] / 200).round().astype("Int64")

    # 競馬場×距離カテゴリ でグループ化して累積勝率・平均着順を計算
    df["_win"] = (df["着順_num"] == 1).astype(float)

    # 競馬場×距離カテゴリ 別（±200m近似）
    grp_jyo_dist = df.groupby(["馬名", "競馬場cd", "距離カテゴリ_jyo"])
    df["競馬場距離過去勝率"] = grp_jyo_dist["_win"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["競馬場距離過去平均着順"] = grp_jyo_dist["着順_num"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # 競馬場別（距離問わず）
    grp_jyo = df.groupby(["馬名", "競馬場cd"])
    df["競馬場過去勝率"] = grp_jyo["_win"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["競馬場過去平均着順"] = grp_jyo["着順_num"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # 不要な一時列を削除
    df = df.drop(columns=["_win", "距離カテゴリ_jyo", "距離_num"], errors="ignore")

    # ── ③ 開催時期（月・季節） ────────────────────────────────────────
    race_date = df["race_id"].astype(str).str[:8]
    df["開催月"] = pd.to_numeric(race_date.str[4:6], errors="coerce")
    df["開催季節"] = pd.cut(
        df["開催月"],
        bins=[0, 3, 6, 9, 12],
        labels=[1, 2, 3, 4],
    ).astype(float)

    return df


def build_features(csv_path="race_data_clean.csv", out_path="race_features.csv"):
    print("データ読み込み中...")
    df = load_and_prepare(csv_path)

    print("過去成績を集計中（時間がかかります）...")
    df = add_horse_history_features(df)

    print("レース内相対特徴量を追加中...")
    df = add_race_relative_features(df)

    print("騎手・調教師の成績を集計中...")
    df = add_jockey_trainer_features(df)

    # 距離カテゴリ
    df["距離カテゴリ"] = pd.cut(
        df["レース番号"], bins=[0, 4, 8, 12], labels=[1, 2, 3]
    ).astype(float)

    # 馬ごとの距離カテゴリ別平均着順
    df = df.sort_values(["馬名", "race_id"]).reset_index(drop=True)
    dist_avg = []
    for _, group in df.groupby(["馬名", "距離カテゴリ"]):
        group = group.copy()
        group["距離別過去平均着順"] = group["着順_num"].shift(1).expanding().mean()
        dist_avg.append(group)
    df = pd.concat(dist_avg).sort_values(["race_id", "馬番"]).reset_index(drop=True)

    print("追加特徴量（競馬場適性・脚質・時期）を生成中...")
    df = add_extra_advanced_features(df)

    # 距離を数値に
    if "距離" in df.columns:
        df["距離"] = pd.to_numeric(df["距離"], errors="coerce")
        df["距離"] = df["距離"].fillna(df["距離"].median())

    FEATURE_COLS = [
        "race_id", "馬名", "着順_num",
        "単勝オッズ",
        "枠番", "馬番", "斤量", "斤量_相対",
        "年齢", "is_male", "is_female", "is_castrated",
        "馬体重", "体重増減", "馬体重_相対",
        "人気",
        "上り", "出走頭数", "競馬場cd", "レース番号",
        "過去出走数", "過去平均着順", "過去勝率", "過去複勝率",
        "過去平均上り", "直近3走平均着順",
        "過去平均タイム秒", "直近3走平均タイム秒", "過去最速タイム秒",
        "直近3走平均上り", "過去平均体重増減",
        "距離カテゴリ", "距離別過去平均着順",
        "騎手勝率", "騎手複勝率", "調教師勝率", "調教師複勝率",
        "距離", "馬場状態_num", "is_turf", "クラス_num",
        "前走間隔",
        "同距離過去勝率", "同距離過去平均着順",
        "良馬場勝率", "重馬場勝率",
        # 上がり関連特徴量
        "過去最速上り", "上り偏差", "距離別過去平均上り",
        # 競馬場×距離 適性
        "競馬場距離過去勝率", "競馬場距離過去平均着順",
        "競馬場過去勝率", "競馬場過去平均着順",
        # 脚質
        "過去平均先行指数", "先行馬フラグ",
        # 開催時期
        "開催月", "開催季節",
        # 前走情報
        "前走着順", "前走上り", "前走距離", "距離変化",
        # 連続好走・成長
        "連続複勝フラグ", "連続勝利フラグ", "近走改善度",
        # タイム差
        "平均タイム差",
        # 騎手×競馬場
        "騎手競馬場勝率",
    ]

    out = df[[c for c in FEATURE_COLS if c in df.columns]]
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"保存完了 → {out_path}（{len(out)}行 × {len(out.columns)}列）")
    return out


if __name__ == "__main__":
    df = build_features()
    print("\nサンプル:")
    print(df.head(3).to_string())