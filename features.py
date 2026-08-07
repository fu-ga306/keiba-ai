import os
import pandas as pd
import numpy as np
import re

# ── 実開催日による時系列整列（2026-07-28修正）────────────────────────────────
# race_idは「年(4)+場(2)+回(2)+日目(2)+R(2)」であって日付ではない。にもかかわらず
# 従来は df.sort_values(["馬名","race_id"]) の順を時系列とみなして「過去走」を
# 決めていた。場コードが開催回より先に並ぶため、例えば1月の中山(場06)は7月の
# 札幌(場01)より後ろに並び、未来のレースが過去成績として集計されていた。
# 実日付で検証したところ2025年だけで馬内レース対の45.7%が逆転していた。
# race_dates.csv（build_race_dates.pyで生成）で実日付を引き、それ順に並べる。
_RACE_DATE_MAP = None


def _race_date_map():
    global _RACE_DATE_MAP
    if _RACE_DATE_MAP is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "race_dates.csv")
        if os.path.exists(p):
            m = pd.read_csv(p, dtype={"kaisai_key": str})
            m["date"] = pd.to_datetime(m["date"], errors="coerce")
            _RACE_DATE_MAP = dict(zip(m["kaisai_key"], m["date"]))
        else:
            _RACE_DATE_MAP = {}
            print("  [警告] race_dates.csv が無いため race_id順で整列します"
                  "（未来のレースが過去成績に混入する既知の不具合）")
    return _RACE_DATE_MAP


def attach_race_date(df):
    """race_id から実開催日 _race_dt を付与（無ければNaT）。"""
    m = _race_date_map()
    if not m:
        df["_race_dt"] = pd.NaT
        return df
    key = df["race_id"].astype(str).str.replace(r"\.0$", "", regex=True).str[:10]
    df["_race_dt"] = key.map(m)
    return df


def sort_by_horse_time(df):
    """馬ごとに時系列順で並べる。日付不明はrace_id順で後ろに置く（予測対象＝最新の想定）。"""
    if "_race_dt" not in df.columns:
        df = attach_race_date(df)
    return df.sort_values(["馬名", "_race_dt", "race_id"],
                          na_position="last").reset_index(drop=True)
from datetime import datetime


# ── 着差・通過ヘルパー ───────────────────────────────────────────────────────
_CHAKUSA_MAP = {
    "同着": 0.0, "ハナ": 0.05, "アタマ": 0.10, "クビ": 0.20,
    "1/2": 0.30, "3/4": 0.35, "大": 5.0,
}

def chakusa_to_sec(s):
    """着差文字列を秒換算（1着=0、NaN→NaN）。1馬身≒0.2秒で計算。"""
    if pd.isna(s) or str(s).strip() == "":
        return np.nan
    s = str(s).strip()
    if s in _CHAKUSA_MAP:
        return _CHAKUSA_MAP[s]
    # "1.1/4", "2.1/2" 形式
    m = re.match(r"^(\d+)\.(\d+)/(\d+)$", s)
    if m:
        lengths = int(m.group(1)) + int(m.group(2)) / int(m.group(3))
        return lengths * 0.2
    # "1/2", "3/4" 形式
    m = re.match(r"^(\d+)/(\d+)$", s)
    if m:
        return int(m.group(1)) / int(m.group(2)) * 0.2
    # "1馬身", "2馬身" / "1 1/2馬身", "2.1/2馬身" 形式
    m = re.match(r"^(\d+(?:\.\d+)?)\s*馬身$", s)
    if m:
        return float(m.group(1)) * 0.2
    m = re.match(r"^(\d+)\s+(\d+)/(\d+)\s*馬身$", s)
    if m:
        lengths = int(m.group(1)) + int(m.group(2)) / int(m.group(3))
        return lengths * 0.2
    # 純数字 "1", "2", ...
    try:
        return float(s) * 0.2
    except Exception:
        return np.nan


def _extract_4corner(passage):
    """通過順位文字列から最終コーナー位置を抽出（"1-2-3-4" → 4.0）。"""
    try:
        if pd.isna(passage) or str(passage).strip() == "":
            return np.nan
        positions = [int(x) for x in str(passage).split("-") if x.isdigit()]
        return float(positions[-1]) if positions else np.nan
    except Exception:
        return np.nan


def _extract_1corner(passage):
    """通過順位文字列から1コーナー位置を抽出（"8-5-3-2" → 8.0）。"""
    try:
        if pd.isna(passage) or str(passage).strip() == "":
            return np.nan
        positions = [int(x) for x in str(passage).split("-") if x.isdigit()]
        return float(positions[0]) if positions else np.nan
    except Exception:
        return np.nan
# ─────────────────────────────────────────────────────────────────────────────


def load_and_prepare(csv_path="race_data_clean.csv", df=None):
    # df が渡されればCSVを読まずにそれを使う（予測時の一本化用）。
    if df is None:
        df = pd.read_csv(csv_path, low_memory=False)
    else:
        df = df.copy()

    def time_to_sec(t):
        try:
            if pd.isna(t) or t == "":
                return np.nan
            parts = str(t).split(":")
            return int(parts[0]) * 60 + float(parts[1])
        except:
            return np.nan

    df["タイム秒"] = df["タイム"].apply(time_to_sec) if "タイム" in df.columns else np.nan

    df["race_id"]   = df["race_id"].astype(str)
    df["年"]        = df["race_id"].str[0:4].astype(int)
    df["競馬場cd"]  = df["race_id"].str[4:6].astype(int)
    df["回"]        = df["race_id"].str[6:8].astype(int)
    df["日"]        = df["race_id"].str[8:10].astype(int)
    df["レース番号"] = df["race_id"].str[10:12].astype(int)

    # 性別・年齢: 予測データには「性齢」(例:牡3)しかなく、縦結合後は履歴由来で
    # 「性別」列は存在するが予測行はNaN。性齢がある行は性齢から性別/年齢を埋める。
    if "性齢" in df.columns:
        seire = df["性齢"].astype(str)
        sex_from_seire = pd.Series(np.nan, index=df.index, dtype=object)
        sex_from_seire[seire.str.contains("牡", na=False)] = "牡"
        sex_from_seire[seire.str.contains("牝", na=False)] = "牝"
        sex_from_seire[seire.str.contains("セ", na=False)] = "セ"
        if "性別" not in df.columns:
            df["性別"] = np.nan
        sei = df["性別"].astype(object)
        need = sei.isna() | sei.astype(str).isin(["nan", "", "None"])
        df["性別"] = sei.where(~need, sex_from_seire)
        age_from_seire = pd.to_numeric(seire.str.extract(r"(\d+)")[0], errors="coerce")
        if "年齢" not in df.columns:
            df["年齢"] = np.nan
        df["年齢"] = pd.to_numeric(df["年齢"], errors="coerce").fillna(age_from_seire)
    if "性別" not in df.columns:
        df["性別"] = ""
    df["性別"] = df["性別"].fillna("")

    df["is_male"]      = (df["性別"] == "牡").astype(int)
    df["is_female"]    = (df["性別"] == "牝").astype(int)
    df["is_castrated"] = (df["性別"] == "セ").astype(int)
    if "人気" in df.columns:
        df["人気_inv"] = 1 / pd.to_numeric(df["人気"], errors="coerce").clip(lower=1)
    else:
        df["人気_inv"] = np.nan

    # 馬体重・体重増減: 予測データは "422kg(+2)" 形式。学習データは数値。なければパースする。
    if "馬体重" in df.columns and df["馬体重"].astype(str).str.contains("kg|\\(", regex=True).any():
        raw = df["馬体重"].astype(str)
        parsed_w = pd.to_numeric(raw.str.extract(r"(\d+)")[0], errors="coerce")
        df["馬体重"] = df["馬体重"].where(~raw.str.contains("kg|\\(", regex=True), parsed_w)
        # 体重増減: "466(+4)" の括弧内を数値化。予測行は 体重増減 列が履歴由来で既に存在し
        # NaN のことが多いため、「馬体重が生テキスト(括弧付き)の行」だけ上書きして埋める。
        # （旧実装は if "体重増減" not in df.columns で、列が存在する予測経路では常にスキップされ
        #   本番の体重増減・体重増減_異常度が全てNaNになっていた。）
        _delta = pd.to_numeric(raw.str.extract(r"\(([+-]?\d+)\)")[0], errors="coerce")
        _has_paren = raw.str.contains(r"\(", regex=True)
        if "体重増減" not in df.columns:
            df["体重増減"] = _delta
        else:
            df["体重増減"] = df["体重増減"].where(~_has_paren, _delta)

    # is_turf / 馬場状態_num / クラス_num: 縦結合後、履歴由来でこれらの列は存在するが、
    # 予測対象行(予測データ由来)は値がNaN。「列の有無」でなく「値がNaN」の行を
    # 元の文字列列(馬場種別/馬場状態/レースクラス)から埋める。
    baba_map = {"良": 1, "稍重": 2, "重": 3, "不良": 4, "稍": 2, "不": 4}
    class_map = {
        "新馬": 1, "未勝利": 2, "1勝クラス": 3, "2勝クラス": 4,
        "3勝クラス": 5, "オープン": 6, "G3": 7, "G2": 8, "G1": 9,
        "500万下": 3, "1000万下": 4, "1600万下": 5,
    }
    if "馬場種別" in df.columns:
        turf_from_str = df["馬場種別"].astype(str).str.contains("芝").astype(float)
        if "is_turf" not in df.columns:
            df["is_turf"] = np.nan
        df["is_turf"] = pd.to_numeric(df["is_turf"], errors="coerce").fillna(turf_from_str)
    if "馬場状態" in df.columns:
        baba_from_str = df["馬場状態"].astype(str).map(baba_map)
        if "馬場状態_num" not in df.columns:
            df["馬場状態_num"] = np.nan
        df["馬場状態_num"] = pd.to_numeric(df["馬場状態_num"], errors="coerce").fillna(baba_from_str)
    if "レースクラス" in df.columns:
        class_from_str = df["レースクラス"].astype(str).map(class_map)
        if "クラス_num" not in df.columns:
            df["クラス_num"] = np.nan
        df["クラス_num"] = pd.to_numeric(df["クラス_num"], errors="coerce").fillna(class_from_str)

    # クラス_num: なければレースクラスから作る（上で処理済みだが念のため）
    if "クラス_num" not in df.columns and "レースクラス" in df.columns:
        df["クラス_num"] = df["レースクラス"].astype(str).map(class_map)

    # 回り_num: 右=1, 左=2, 直=3
    mawari_map = {"右": 1, "左": 2, "直": 3}
    if "回り" in df.columns:
        if "回り_num" not in df.columns:
            df["回り_num"] = np.nan
        df["回り_num"] = pd.to_numeric(df["回り_num"], errors="coerce").fillna(
            df["回り"].astype(str).map(mawari_map)
        )

    # 数値であるべき列を確実に数値化（予測データは文字列のことがあり、
    # 文字列のままだと groupby集計で連結され "54.057.0..." のように壊れる）。
    for col in ["斤量", "馬番", "枠番", "距離", "人気", "単勝オッズ",
                "馬体重", "体重増減", "年齢", "馬場状態_num", "クラス_num",
                "着順_num", "上り", "賞金"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _loop_features_per_horse(df):
    """ループが計算する全特徴量を per-horse で再現（値はループと完全一致を目指す）。"""
    df = sort_by_horse_time(df)
    n = len(df)
    # 事前計算列（ループ内で valid[...] として参照される生データ）
    chaku = pd.to_numeric(df["着順_num"], errors="coerce").to_numpy()
    agari = pd.to_numeric(df.get("上り"), errors="coerce").to_numpy() if "上り" in df.columns else np.full(n, np.nan)
    taim  = pd.to_numeric(df.get("タイム秒"), errors="coerce").to_numpy() if "タイム秒" in df.columns else np.full(n, np.nan)
    dist  = pd.to_numeric(df.get("距離"), errors="coerce").to_numpy() if "距離" in df.columns else np.full(n, np.nan)
    taiju = pd.to_numeric(df.get("体重増減"), errors="coerce").to_numpy() if "体重増減" in df.columns else np.full(n, np.nan)
    clsn  = pd.to_numeric(df.get("クラス_num"), errors="coerce").to_numpy() if "クラス_num" in df.columns else np.full(n, np.nan)
    turf  = pd.to_numeric(df.get("is_turf"), errors="coerce").to_numpy() if "is_turf" in df.columns else np.full(n, np.nan)
    raceno = pd.to_numeric(df.get("レース番号"), errors="coerce").to_numpy() if "レース番号" in df.columns else np.full(n, np.nan)
    baba  = df.get("馬場状態").astype(str).to_numpy() if "馬場状態" in df.columns else np.array([""]*n)
    rid_str = df["race_id"].astype(str).to_numpy()
    if "_race_dt" not in df.columns:
        df = attach_race_date(df)
    race_dt = df["_race_dt"].to_numpy()   # 前走間隔は実開催日で計算する
    # 着差→秒（ループの valid.apply(lambda r: 0 if 着順==1 else chakusa_to_sec(着差)) と同じ）
    if "着差" in df.columns:
        cs_raw = df["着差"].map(chakusa_to_sec).to_numpy()
        chakusa_sec = np.where(chaku == 1, 0.0, cs_raw)
    else:
        chakusa_sec = np.full(n, np.nan)
    # 通過→4角/1角
    if "通過" in df.columns:
        c4 = df["通過"].map(_extract_4corner).to_numpy(dtype=float)
        c1 = df["通過"].map(_extract_1corner).to_numpy(dtype=float)
    else:
        c4 = np.full(n, np.nan); c1 = np.full(n, np.nan)
    # 平均タイム差用の parse_chakusa（ループ 629-637 と同じ: "同着"/"大差"/空→NaN, else float）
    def parse_chakusa(x):
        try:
            if pd.isna(x) or str(x).strip() in ["", "同着", "大差"]:
                return np.nan
            return float(str(x))
        except Exception:
            return np.nan
    cs2 = df["着差"].map(parse_chakusa).to_numpy(dtype=float) if "着差" in df.columns else np.full(n, np.nan)

    COLS = ["過去出走数","過去平均着順","過去勝率","過去複勝率","過去平均上り","直近3走平均着順",
            "過去平均タイム秒","直近3走平均タイム秒","過去最速タイム秒","直近3走平均上り","過去平均体重増減",
            "過去最速上り","上り偏差","前走間隔","同距離過去勝率","同距離過去平均着順","良馬場勝率","重馬場勝率",
            "距離別過去平均上り","前走着順","前走上り","前走距離","距離変化","クラス変化","前走着差_秒","過去平均着差_秒",
            "前走4角位置","過去平均4角位置","前走脚質指数","近5走着差_std","重賞出走フラグ","過去重賞出走数",
            "経験最長距離","経験範囲超過","延長×距離経験不足","長距離複勝率","前走余力","延長×前走余力",
            "連続複勝フラグ","連続勝利フラグ","直近5走勝利数","直近5走複勝数","直近5走平均着順",
            "過去着順_std","直近5走着順_std","直近5走着外率","近走改善度","平均タイム差"]
    res = {c: np.full(n, np.nan) for c in COLS}

    def nanmean(a): return np.nan if len(a)==0 else np.nanmean(a) if np.isfinite(a).any() else np.nan
    def nanstd(a):
        b = a[np.isfinite(a)]
        return b.std(ddof=1) if len(b) >= 2 else np.nan

    order = df.index.to_numpy()
    for _, idxs in df.groupby("馬名").groups.items():
        gi = np.array(idxs)
        k = len(gi)
        for pos in range(k):
            i = gi[pos]
            past = gi[:pos]                      # その馬の過去行(グローバルindex)
            vch = chaku[past]                    # 過去着順(NaN含む)
            vmask = np.isfinite(vch)             # valid = past.dropna(着順)
            v = past[vmask]                      # valid の index
            if len(v) == 0:
                res["過去出走数"][i] = 0
                continue
            vch2 = chaku[v]
            res["過去出走数"][i]      = len(v)
            res["過去平均着順"][i]    = vch2.mean()
            res["過去勝率"][i]        = (vch2 == 1).mean()
            res["過去複勝率"][i]      = (vch2 <= 3).mean()
            res["過去平均上り"][i]    = nanmean(agari[v])
            res["直近3走平均着順"][i] = chaku[v[-3:]].mean()
            res["過去平均タイム秒"][i]    = nanmean(taim[v])
            res["直近3走平均タイム秒"][i] = nanmean(taim[v[-3:]])
            res["過去最速タイム秒"][i]    = np.nan if not np.isfinite(taim[v]).any() else np.nanmin(taim[v])
            res["直近3走平均上り"][i] = nanmean(agari[v[-3:]])
            res["過去平均体重増減"][i]= nanmean(taiju[v])
            res["過去最速上り"][i]    = np.nan if not np.isfinite(agari[v]).any() else np.nanmin(agari[v])
            res["上り偏差"][i]        = nanstd(agari[v])
            # 前走間隔（週）: 実開催日の差。2026-07-28まで race_id[:8] を日付として
            # 解釈していたが、race_idは「年+場+回+日目」で日付ではないため、
            # 実際には「場コードの差」を週数と呼んでいた（例: 札幌1回2日目→2026-01-01）。
            # ※ numpy.timedelta64 に .days は無い（datetime.timedeltaと違う）。
            #   np.timedelta64(1,"D") で割って日数にする。
            d1 = race_dt[i]
            d2 = race_dt[past[-1]]
            if np.isnat(d1) or np.isnat(d2):
                res["前走間隔"][i] = np.nan
            else:
                res["前走間隔"][i] = ((d1 - d2) / np.timedelta64(1, "D")) / 7
            # 同距離(±200m)
            cd = dist[i]
            if np.isfinite(cd):
                sd = v[(dist[v] >= cd-200) & (dist[v] <= cd+200)]
                if len(sd) > 0:
                    res["同距離過去勝率"][i]     = (chaku[sd] == 1).mean()
                    res["同距離過去平均着順"][i] = chaku[sd].mean()
            # 馬場状態別
            ry = v[baba[v] == "良"]; om = v[np.isin(baba[v], ["重","不良"])]
            if len(ry) > 0: res["良馬場勝率"][i] = (chaku[ry] == 1).mean()
            if len(om) > 0: res["重馬場勝率"][i] = (chaku[om] == 1).mean()
            # 距離別過去平均上り（レース番号カテゴリ）
            crn = raceno[i]
            if np.isfinite(crn):
                if crn <= 4: sc = v[raceno[v] <= 4]
                elif crn <= 8: sc = v[(raceno[v] >= 5) & (raceno[v] <= 8)]
                else: sc = v[raceno[v] >= 9]
                if len(sc) > 0: res["距離別過去平均上り"][i] = nanmean(agari[sc])
            # 前走情報（valid最終）
            pv = v[-1]
            res["前走着順"][i] = chaku[pv]
            res["前走上り"][i] = agari[pv]
            res["前走距離"][i] = dist[pv]
            if np.isfinite(cd) and np.isfinite(dist[pv]): res["距離変化"][i] = cd - dist[pv]
            if np.isfinite(clsn[i]) and np.isfinite(clsn[pv]): res["クラス変化"][i] = clsn[i] - clsn[pv]
            res["前走着差_秒"][i] = chakusa_sec[pv]
            res["過去平均着差_秒"][i] = nanmean(chakusa_sec[v])
            # 4角
            res["前走4角位置"][i] = c4[pv]
            if np.isfinite(c1[pv]) and np.isfinite(c4[pv]): res["前走脚質指数"][i] = c1[pv] - c4[pv]
            if np.isfinite(c4[v]).any(): res["過去平均4角位置"][i] = nanmean(c4[v])
            # 近5走着差std
            cs5 = chakusa_sec[v]; cs5 = cs5[np.isfinite(cs5)][-5:]
            res["近5走着差_std"][i] = cs5.std(ddof=1) if len(cs5) >= 2 else np.nan
            # 重賞
            gr = clsn[v]; res["重賞出走フラグ"][i] = float((gr >= 6).any()); res["過去重賞出走数"][i] = float((gr >= 6).sum())
            # 距離適性
            pdists = dist[v][np.isfinite(dist[v])]
            if len(pdists) > 0 and np.isfinite(cd):
                mx = pdists.max()
                res["経験最長距離"][i] = mx
                res["経験範囲超過"][i] = max(cd - mx, 0)
                pd_ = res["前走距離"][i]
                encho = max(cd - (pd_ if np.isfinite(pd_) else cd), 0)
                res["延長×距離経験不足"][i] = encho * (1.0 if cd > mx else 0.0)
                lp = v[dist[v] >= 2001]
                if len(lp) > 0: res["長距離複勝率"][i] = (chaku[lp] <= 3).mean()
            # 前走余力
            pc = chaku[pv]
            yory = (1.0 if pc == 1 else (0.5 if pc <= 3 else 0.0)) if np.isfinite(pc) else np.nan
            res["前走余力"][i] = yory
            pdd = res["前走距離"][i]
            if np.isfinite(cd) and np.isfinite(pdd) and np.isfinite(yory):
                res["延長×前走余力"][i] = max(cd - pdd, 0) * yory
            # 連続フラグ
            if len(v) >= 2:
                l2 = chaku[v[-2:]]
                res["連続複勝フラグ"][i] = 1.0 if (l2 <= 3).all() else 0.0
                res["連続勝利フラグ"][i] = 1.0 if (l2 == 1).all() else 0.0
            # 直近5走
            l5 = chaku[v[-5:]]
            res["直近5走勝利数"][i] = int((l5 == 1).sum())
            res["直近5走複勝数"][i] = int((l5 <= 3).sum())
            res["直近5走平均着順"][i] = l5.mean()
            res["過去着順_std"][i]   = vch2.std(ddof=1) if len(v) >= 2 else np.nan
            res["直近5走着順_std"][i] = l5.std(ddof=1) if len(l5) >= 2 else np.nan
            res["直近5走着外率"][i]  = float((l5 > 3).mean())
            # 近走改善度（len(valid)>=4）
            if len(v) >= 4:
                res["近走改善度"][i] = vch2.mean() - chaku[v[-3:]].mean()
            # 平均タイム差
            cv = cs2[v]; cv = cv[np.isfinite(cv)]
            res["平均タイム差"][i] = cv.mean() if len(cv) > 0 else np.nan
    # 6列(クラス変化/重賞/4角系)はループが「過去出走数==0」で上書きしない→上部の値を残す
    SIX = ["クラス変化","重賞出走フラグ","過去重賞出走数","前走4角位置","過去平均4角位置","前走脚質指数"]
    mask = res["過去出走数"] > 0
    for c in COLS:
        if c in SIX and c in df.columns:
            df[c] = np.where(mask, res[c], pd.to_numeric(df[c], errors="coerce").to_numpy())
        else:
            df[c] = res[c]
    return df


def add_horse_history_features(df):
    """各馬の過去成績を特徴量として追加（当日データ混入なし）。

    ※高速化版: 旧実装は g[col].transform(lambda x: x.shift(1).expanding().mean()) を
      約20回使い、馬ごとにPythonラムダを呼ぶため 40k行で約8分かかっていた（新馬戦の
      履歴フォールバックでハングの主因）。値は完全に保ったまま、群内 shift(1) →
      grouped expanding/rolling（pandasのC実装）に置き換えて 20倍以上 高速化した。
    """
    df = sort_by_horse_time(df)
    g = df.groupby("馬名", sort=False)
    names = df["馬名"]
    df["_win"]  = (df["着順_num"] == 1).astype(float)
    df["_top3"] = (df["着順_num"] <= 3).astype(float)

    def _pe(col, how, k=None):
        """群内 shift(1) 後に grouped expanding/rolling を適用。
        旧 g[col].transform(lambda x: x.shift(1).<op>()) と数値的に完全同一で高速。"""
        s = g[col].shift(1).groupby(names, sort=False)
        if   how == "count": r = s.expanding().count()
        elif how == "mean":  r = s.expanding().mean()
        elif how == "sum":   r = s.expanding().sum()
        elif how == "min":   r = s.expanding().min()
        elif how == "max":   r = s.expanding().max()
        elif how == "std":   r = s.expanding().std()
        elif how == "rmean": r = s.rolling(k, min_periods=1).mean()
        elif how == "rstd":  r = s.rolling(k, min_periods=2).std()
        # grouped結果は群順のMultiIndex。df.index順に並べ直して numpy配列で返す。
        # ※Seriesのまま df[col]=... すると index不一致で1要素ずつ挿入され激遅(8000行×19列=15万回)。
        return r.reset_index(level=0, drop=True).reindex(df.index).to_numpy()

    df["過去出走数"]       = _pe("着順_num", "count")
    df["過去平均着順"]     = _pe("着順_num", "mean")
    df["過去勝率"]         = _pe("_win", "mean")
    df["過去複勝率"]       = _pe("_top3", "mean")
    df["直近3走平均着順"]  = _pe("着順_num", "rmean", 3)
    df["過去平均上り"]     = _pe("上り", "mean")
    df["直近3走平均上り"]  = _pe("上り", "rmean", 3)
    df["過去最速上り"]     = _pe("上り", "min")
    df["上り偏差"]         = _pe("上り", "std")
    df["過去平均体重増減"] = _pe("体重増減", "mean")
    # ── 過去獲得賞金（B-4: 実力指標）─────────────────────────────────
    if "賞金" in df.columns:
        df["過去獲得賞金累計"] = _pe("賞金", "sum")
        df["過去平均獲得賞金"] = _pe("賞金", "mean")
    else:
        df["過去獲得賞金累計"] = np.nan
        df["過去平均獲得賞金"] = np.nan
    # ── ⑥ 馬体重変化の個体差特徴量（今回の増減がその馬にとって異常かをzスコア化）──
    df["体重増減_過去標準偏差"] = _pe("体重増減", "std")
    df["体重増減_異常度"] = (
        (df["体重増減"] - df["過去平均体重増減"])
        / df["体重増減_過去標準偏差"].replace(0, np.nan)
    ).clip(-5, 5)

    # 前走情報
    df["前走着順"] = g["着順_num"].shift(1)
    df["前走上り"] = g["上り"].shift(1)

    # クラス変化（昇降級: 正=昇級、負=降級）
    if "クラス_num" in df.columns:
        _prev_cls = g["クラス_num"].shift(1)
        df["クラス変化"] = pd.to_numeric(df["クラス_num"], errors="coerce") - _prev_cls

    # 前走着差（1着=0、それ以外は着差テキスト→秒換算）※axis=1 apply を map でベクトル化
    if "着差" in df.columns:
        df["_chakusa_sec"] = np.where(
            df["着順_num"] == 1, 0.0, df["着差"].map(chakusa_to_sec)
        )
        df["前走着差_秒"]     = g["_chakusa_sec"].shift(1)
        df["過去平均着差_秒"] = _pe("_chakusa_sec", "mean")
        df["近5走平均着差_秒"] = _pe("_chakusa_sec", "rmean", 5)
    else:
        df["前走着差_秒"]      = np.nan
        df["過去平均着差_秒"]  = np.nan
        df["近5走平均着差_秒"] = np.nan

    # 芝ダート変更フラグ（前走と今回で馬場種別が変わった）
    if "is_turf" in df.columns:
        _prev_turf = g["is_turf"].shift(1)
        df["芝ダート変更"] = (pd.to_numeric(df["is_turf"], errors="coerce") != _prev_turf).astype(float)
        df.loc[_prev_turf.isna(), "芝ダート変更"] = np.nan
    else:
        df["芝ダート変更"] = np.nan

    # 着差安定度（近5走の着差標準偏差）
    if "着差" in df.columns:
        df["近5走着差_std"] = _pe("_chakusa_sec", "rstd", 5)
    else:
        df["近5走着差_std"] = np.nan

    # 重賞実績（クラス_num >= 6 = G3以上の出走歴）
    if "クラス_num" in df.columns:
        df["_is_graded"] = (pd.to_numeric(df["クラス_num"], errors="coerce") >= 6).astype(float)
        df["重賞出走フラグ"]  = _pe("_is_graded", "max")
        df["過去重賞出走数"]  = _pe("_is_graded", "sum")
    else:
        df["重賞出走フラグ"] = np.nan
        df["過去重賞出走数"] = np.nan

    # 前走4角位置・脚質指数（1角-4角: 正=追い込み、負=先行失速）
    if "通過" in df.columns:
        df["_4角位置"] = df["通過"].apply(_extract_4corner)
        df["_1角位置"] = df["通過"].apply(_extract_1corner)
        df["前走4角位置"]   = g["_4角位置"].transform(lambda x: x.shift(1))
        df["過去平均4角位置"] = g["_4角位置"].transform(lambda x: x.shift(1).expanding().mean())
        df["前走脚質指数"]   = g["_1角位置"].transform(lambda x: x.shift(1)) - df["前走4角位置"]
    else:
        df["前走4角位置"]   = np.nan
        df["過去平均4角位置"] = np.nan
        df["前走脚質指数"]   = np.nan

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

    # ── 距離変化系の新特徴量（長距離・距離延長の弱点対策）──
    _cur_dist = pd.to_numeric(df["距離"], errors="coerce")
    # 距離変化比率（今回/前走、1.0超=延長）
    df["距離変化比率"] = _cur_dist / df["前走距離"].replace(0, np.nan)
    # 大幅延長フラグ（200m超の延長）
    df["大幅延長フラグ"] = ((df["距離変化"] > 200)).astype(float)
    # 大幅短縮フラグ（200m超の短縮）
    df["大幅短縮フラグ"] = ((df["距離変化"] < -200)).astype(float)
    # 延長幅（延長のみ正値、短縮は0）
    df["距離延長幅"] = df["距離変化"].clip(lower=0)
    # 長距離フラグ（今回が2001m以上）
    df["長距離フラグ"] = (_cur_dist >= 2001).astype(float)
    # 大幅延長 × 長距離（初挑戦の長距離延長＝最も飛びやすい）
    df["大幅延長×長距離"] = df["大幅延長フラグ"] * df["長距離フラグ"]
    # ※「距離延長×先行」は先行馬フラグ生成後に作る（add_pace_features内）

    df = df.drop(columns=["_win", "_top3", "_last3_avg", "_all_avg"], errors="ignore")

    # ── ループ撤去: per-horse numpy で同値高速化（ゴールデン全列一致検証済み） ──
    df = _loop_features_per_horse(df)
    return df


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
        # レース内ランク（1=最良。NaNは最下位扱い）
        if "過去勝率" in group.columns:
            group["レース内_過去勝率ランク"] = group["過去勝率"].rank(
                ascending=False, method="min", na_option="bottom")
        if "直近3走平均着順" in group.columns:
            group["レース内_直近3走平均着順ランク"] = group["直近3走平均着順"].rank(
                ascending=True, method="min", na_option="bottom")
        if "過去平均上り" in group.columns:
            group["レース内_過去平均上りランク"] = group["過去平均上り"].rank(
                ascending=True, method="min", na_option="bottom")
        if "騎手勝率" in group.columns:
            group["レース内_騎手勝率ランク"] = group["騎手勝率"].rank(
                ascending=False, method="min", na_option="bottom")
        rel_rows.append(group)
    return pd.concat(rel_rows, ignore_index=True)


def add_jockey_trainer_features(df):
    """騎手・調教師・馬主の累積勝率を特徴量として追加（リーク防止：その時点までのデータのみ）。
    生の累積勝率に加え、ベイズスムージング版（少数サンプルを事前平均へ収縮）も付与する。"""
    df = df.sort_values(["race_id"]).reset_index(drop=True)

    # ベイズスムージング設定（少数出走のカテゴリを全体平均へ収縮）
    K_SMOOTH   = 20      # 収縮強度（この出走数で事前:実測=1:1）
    PRIOR_WIN  = 0.075   # 全体勝率の事前値（≒1/平均頭数）
    PRIOR_FUKU = 0.22    # 全体複勝率の事前値（≒3/平均頭数）

    has_owner = "馬主" in df.columns

    jockey_stats  = {}
    trainer_stats = {}
    owner_stats   = {}
    jockey_winrate  = []
    jockey_fukusho  = []
    trainer_winrate = []
    trainer_fukusho = []
    jockey_win_sm   = []
    trainer_win_sm  = []
    owner_winrate   = []
    owner_fukusho   = []
    owner_win_sm    = []

    def _sm(wins, runs, prior):
        """ベイズスムージング: (wins + K*prior) / (runs + K)。runs=0でもpriorを返す。"""
        return (wins + K_SMOOTH * prior) / (runs + K_SMOOTH)

    for _, row in df.iterrows():
        jockey   = row["騎手"]
        trainer  = row["調教師"]
        owner    = row["馬主"] if has_owner else None
        chakujun = row["着順_num"]

        if jockey not in jockey_stats:
            jockey_stats[jockey] = {"runs": 0, "wins": 0, "top3": 0}
        if trainer not in trainer_stats:
            trainer_stats[trainer] = {"runs": 0, "wins": 0, "top3": 0}
        if has_owner and owner not in owner_stats:
            owner_stats[owner] = {"runs": 0, "wins": 0, "top3": 0}

        js = jockey_stats[jockey]
        ts = trainer_stats[trainer]
        jockey_winrate.append(js["wins"]  / js["runs"] if js["runs"] > 0 else np.nan)
        jockey_fukusho.append(js["top3"]  / js["runs"] if js["runs"] > 0 else np.nan)
        trainer_winrate.append(ts["wins"] / ts["runs"] if ts["runs"] > 0 else np.nan)
        trainer_fukusho.append(ts["top3"] / ts["runs"] if ts["runs"] > 0 else np.nan)
        jockey_win_sm.append(_sm(js["wins"], js["runs"], PRIOR_WIN))
        trainer_win_sm.append(_sm(ts["wins"], ts["runs"], PRIOR_WIN))
        if has_owner:
            os_ = owner_stats[owner]
            owner_winrate.append(os_["wins"] / os_["runs"] if os_["runs"] > 0 else np.nan)
            owner_fukusho.append(os_["top3"] / os_["runs"] if os_["runs"] > 0 else np.nan)
            owner_win_sm.append(_sm(os_["wins"], os_["runs"], PRIOR_WIN))

        if not np.isnan(chakujun):
            jockey_stats[jockey]["runs"]   += 1
            trainer_stats[trainer]["runs"] += 1
            if has_owner:
                owner_stats[owner]["runs"] += 1
            if chakujun == 1:
                jockey_stats[jockey]["wins"]   += 1
                trainer_stats[trainer]["wins"] += 1
                if has_owner:
                    owner_stats[owner]["wins"] += 1
            if chakujun <= 3:
                jockey_stats[jockey]["top3"]   += 1
                trainer_stats[trainer]["top3"] += 1
                if has_owner:
                    owner_stats[owner]["top3"] += 1

    df["騎手勝率"]   = jockey_winrate
    df["騎手複勝率"] = jockey_fukusho
    df["調教師勝率"]   = trainer_winrate
    df["調教師複勝率"] = trainer_fukusho
    df["騎手勝率_sm"]   = jockey_win_sm
    df["調教師勝率_sm"] = trainer_win_sm
    if has_owner:
        df["馬主勝率"]    = owner_winrate
        df["馬主複勝率"]  = owner_fukusho
        df["馬主勝率_sm"] = owner_win_sm
    else:
        df["馬主勝率"] = np.nan
        df["馬主複勝率"] = np.nan
        df["馬主勝率_sm"] = np.nan

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

    # ── 騎手の直近トレンド（「今乗れている騎手」を捉える） ──────────────
    # 直近30走の勝率・複勝率を追跡（通算とは別に「最近の調子」を見る）
    from collections import deque
    df3 = df.sort_values("race_id").reset_index(drop=True)
    jockey_recent = {}  # {騎手: deque(maxlen=30) of (win, top3)}
    recent_win  = []
    recent_fuku = []
    recent_trend = []  # 直近10走 vs その前20走 の勝率差（調子の上昇/下降）

    for _, row in df3.iterrows():
        jockey   = row["騎手"]
        chakujun = row["着順_num"]

        if jockey not in jockey_recent:
            jockey_recent[jockey] = deque(maxlen=30)
        dq = jockey_recent[jockey]

        # 現時点での直近成績（過去データのみ）
        if len(dq) > 0:
            wins = sum(w for w, _ in dq)
            top3 = sum(t for _, t in dq)
            recent_win.append(wins / len(dq))
            recent_fuku.append(top3 / len(dq))
            # 調子トレンド: 直近10走の勝率 - 直近11〜30走の勝率
            recent10 = list(dq)[-10:]
            older    = list(dq)[:-10]
            if len(recent10) >= 5 and len(older) >= 5:
                w_recent = sum(w for w, _ in recent10) / len(recent10)
                w_older  = sum(w for w, _ in older) / len(older)
                recent_trend.append(w_recent - w_older)
            else:
                recent_trend.append(np.nan)
        else:
            recent_win.append(np.nan)
            recent_fuku.append(np.nan)
            recent_trend.append(np.nan)

        # 結果を追加（着順確定後）
        if not np.isnan(chakujun):
            dq.append((1 if chakujun == 1 else 0, 1 if chakujun <= 3 else 0))

    df3["騎手直近勝率"]     = recent_win
    df3["騎手直近複勝率"]   = recent_fuku
    df3["騎手調子トレンド"] = recent_trend
    df = df.merge(
        df3[["race_id", "馬名", "騎手直近勝率", "騎手直近複勝率", "騎手調子トレンド"]],
        on=["race_id", "馬名"], how="left")

    # レース内_騎手勝率ランク（騎手勝率付与後にここで計算）
    if "騎手勝率" in df.columns:
        df["レース内_騎手勝率ランク"] = df.groupby("race_id")["騎手勝率"].rank(
            ascending=False, method="min", na_option="bottom")

    return df


def add_extra_advanced_features(df):
    """
    精度向上のための追加特徴量（高速化版）：
    ① 競馬場×距離 過去成績（groupby一括処理）
    ② 脚質（先行/差し/追込）
    ③ 開催時期（月・季節）
    """
    df = sort_by_horse_time(df)

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

        # 距離延長 × 先行（逃げ先行馬の距離延長は前半飛ばしてバテやすい＝弱点対策）
        if "距離延長幅" in df.columns:
            df["距離延長×先行"] = df["距離延長幅"].fillna(0) * df["先行馬フラグ"].fillna(0)

        # ── ⑤ ペース想定特徴量（レース内の先行馬構成から想定ペースを推定） ──
        # レース内で「先行馬フラグ=1」の馬が何頭・何割いるか集計。
        # 先行馬が多いほどハイペースになりやすく、差し馬に有利な展開を示唆する。
        race_pace = df.groupby("race_id")["先行馬フラグ"].agg(
            想定先行馬数="sum", 出走頭数_tmp="count"
        ).reset_index()
        race_pace["想定先行馬率"] = race_pace["想定先行馬数"] / race_pace["出走頭数_tmp"].replace(0, np.nan)
        df = df.merge(race_pace[["race_id", "想定先行馬数", "想定先行馬率"]], on="race_id", how="left")

        # 各馬にとって「自分以外の先行馬がどれだけいるか」（差し馬視点での恩恵度）
        df["他馬想定先行馬数"] = df["想定先行馬数"] - df["先行馬フラグ"].fillna(0)
        # 差し馬(先行馬フラグ=0)が、他に先行馬が多いレースに出る場合に有利
        df["差し馬×ハイペース想定"] = (
            (df["先行馬フラグ"] == 0).astype(float) * df["他馬想定先行馬数"]
        )
        # ── ⑥ 精密展開特徴（2026-07-27・通過順位ベース）──────────────────
        # 旧ペース系(先行馬フラグ/想定先行馬数等)は重要度ほぼゼロだった。原因は
        # 「1角位置の生値を頭数/3で二値化」の粗さ。頭数で正規化した連続値で作り直す。
        # 全てレース前情報のみ（脚質は過去走のshift、レース集計は出走馬の過去脚質）。
        _n = pd.to_numeric(df["出走頭数"], errors="coerce")
        df["_相対位置"] = df["先行指数"] / _n.replace(0, np.nan)   # 当該レースの値(shift用)
        _g = df.groupby("馬名")["_相対位置"]
        # 脚質スコア: 近3走の相対1角位置(0=最前, 1=最後方)。前走ほど重視は rolling で近似
        df["脚質スコア"] = _g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        df["前走相対位置"] = _g.transform(lambda x: x.shift(1))
        # 先行力(0-1連続): 前に行く気質の強さ。逃げ気質=平均して先頭付近
        df["_先行力"] = (0.5 - df["脚質スコア"]).clip(lower=0) * 2.0
        df["_逃げ気質"] = (df["脚質スコア"] <= 0.15).astype(float)
        # レース内集計(出走馬の過去脚質から想定ペースを連続値で)
        _rp = df.groupby("race_id").agg(
            想定逃げ馬数=("_逃げ気質", "sum"), 先行圧=("_先行力", "sum")).reset_index()
        df = df.merge(_rp, on="race_id", how="left")
        _other = (df["先行圧"] - df["_先行力"].fillna(0)).clip(lower=0)
        # 差し馬(スコア大)×他馬の先行圧が高い=ハイペース恩恵。連続×連続で旧二値版を置換
        df["差し×先行圧"] = df["脚質スコア"].fillna(0.5) * _other
        # 単騎逃げ=楽逃げ候補 / 逃げ争い=同型競合で共倒れリスク
        df["単騎逃げ"] = df["_逃げ気質"] * (df["想定逃げ馬数"] == 1).astype(float)
        df["逃げ争い"] = df["_逃げ気質"] * (df["想定逃げ馬数"] - 1).clip(lower=0)
        # コース×脚質バイアス: この競馬場×トラック×距離帯で自分の脚質の複勝率。
        # レース単位で「そのレースより前」だけをcumsumで集計(同一レース内リーク無し)
        df["_脚質bin"] = pd.cut(df["脚質スコア"], bins=[-0.01, 0.2, 0.4, 0.65, 1.01],
                                labels=[1, 2, 3, 4]).astype(float)
        if "競馬場cd" not in df.columns:
            df["競馬場cd"] = df["race_id"].astype(str).str[4:6].astype(int)
        _turf = df["馬場"].astype(str).str.contains("芝").astype(int) if "馬場" in df.columns else 0
        df["_track"] = _turf
        df["_距離帯"] = (pd.to_numeric(df["距離"], errors="coerce") / 400).round()
        df["_fuku3"] = (df["着順_num"] <= 3).astype(float)
        _keys = ["競馬場cd", "_track", "_距離帯", "_脚質bin"]
        _t = (df.dropna(subset=["_脚質bin", "_距離帯"])
                .groupby(_keys + ["race_id"])["_fuku3"].agg(["sum", "count"]).reset_index()
                .sort_values("race_id"))
        _gk = _t.groupby(_keys)
        _t["_cs"] = _gk["sum"].cumsum() - _t["sum"]      # 自レースを除いた過去合計
        _t["_cc"] = _gk["count"].cumsum() - _t["count"]
        _t["コース脚質バイアス"] = np.where(_t["_cc"] >= 30, _t["_cs"] / _t["_cc"], np.nan)
        df = df.merge(_t[_keys + ["race_id", "コース脚質バイアス"]],
                      on=_keys + ["race_id"], how="left")
        # 学習ビルド(大データ)では最新累積値をCSV保存し、予測時(履歴を対象馬に絞るため
        # コース全体の集計が組めない)はCSVから補完する(course_bias.csvと同じ流儀)。
        _bias_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "course_style_bias.csv")
        if len(df) > 100000:
            _latest = _t.sort_values("race_id").groupby(_keys).tail(1).copy()
            _tot_c = _latest["_cc"] + _latest["count"]
            _latest["コース脚質バイアス"] = np.where(
                _tot_c >= 30, (_latest["_cs"] + _latest["sum"]) / _tot_c, np.nan)
            _latest.dropna(subset=["コース脚質バイアス"])[_keys + ["コース脚質バイアス"]].to_csv(
                _bias_csv, index=False, encoding="utf-8-sig")
        elif os.path.exists(_bias_csv):
            try:
                _lk = pd.read_csv(_bias_csv).rename(columns={"コース脚質バイアス": "_bias_lk"})
                for _k in _keys:
                    _lk[_k] = _lk[_k].astype(df[_k].dtype)
                df = df.merge(_lk, on=_keys, how="left")
                df["コース脚質バイアス"] = df["コース脚質バイアス"].fillna(df["_bias_lk"])
                df = df.drop(columns=["_bias_lk"], errors="ignore")
                # 予測は1レース分しか無いため脚質binが偏り、上の完全一致では
                # 25〜89%が埋まらない（2026-08-01に検出）。粗い鍵へ段階的に落とす。
                _lk2 = pd.read_csv(_bias_csv)
                for _sub in (["競馬場cd", "_track", "_距離帯"],
                             ["競馬場cd", "_track"], ["_track"]):
                    if df["コース脚質バイアス"].isna().sum() == 0:
                        break
                    _agg = (_lk2.groupby(_sub)["コース脚質バイアス"].mean()
                            .rename("_bias_fb").reset_index())
                    for _k in _sub:
                        _agg[_k] = _agg[_k].astype(df[_k].dtype)
                    df = df.merge(_agg, on=_sub, how="left")
                    df["コース脚質バイアス"] = df["コース脚質バイアス"].fillna(df["_bias_fb"])
                    df = df.drop(columns=["_bias_fb"], errors="ignore")
            except Exception as _e:
                print(f"  [コース脚質バイアス] CSV補完スキップ: {_e}")
        df = df.drop(columns=["_相対位置", "_先行力", "_逃げ気質", "_脚質bin",
                              "_track", "_距離帯", "_fuku3"], errors="ignore")
    else:
        df["過去平均先行指数"] = np.nan
        df["先行馬フラグ"]     = np.nan
        df["想定先行馬数"]     = np.nan
        df["想定先行馬率"]     = np.nan
        df["他馬想定先行馬数"] = np.nan
        df["差し馬×ハイペース想定"] = np.nan
        for _c in ["脚質スコア", "前走相対位置", "想定逃げ馬数", "先行圧",
                   "差し×先行圧", "単騎逃げ", "逃げ争い", "コース脚質バイアス"]:
            df[_c] = np.nan

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

    # 回り別_過去勝率・過去複勝率（右/左/直 それぞれの実績）
    if "回り_num" in df.columns:
        df["_fuku"] = (df["着順_num"] <= 3).astype(float)
        grp_mawari = df.groupby(["馬名", "回り_num"])
        df["回り別_過去勝率"]  = grp_mawari["_win"].transform(lambda x: x.shift(1).expanding().mean())
        df["回り別_過去複勝率"] = grp_mawari["_fuku"].transform(lambda x: x.shift(1).expanding().mean())
        df = df.drop(columns=["_fuku"], errors="ignore")

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

    # ── ④ コース形態の静的ルックアップ（競馬場固定値）─────────────────────
    # 競馬場cd → (直線長_m, 坂あり) 代表値（外回り/芝コースの値）
    _JYO_COURSE = {
        1: (264, 0),   # 札幌
        2: (260, 0),   # 函館
        3: (292, 0),   # 福島
        4: (659, 0),   # 新潟（外回り）
        5: (526, 1),   # 東京
        6: (310, 1),   # 中山（急坂）
        7: (413, 1),   # 中京
        8: (404, 1),   # 京都（外回り）
        9: (474, 1),   # 阪神（外回り）
        10: (293, 0),  # 小倉
    }
    if "競馬場cd" in df.columns:
        cd = pd.to_numeric(df["競馬場cd"], errors="coerce")
        df["直線長_m"] = cd.map({k: v[0] for k, v in _JYO_COURSE.items()})
        df["坂あり"]   = cd.map({k: v[1] for k, v in _JYO_COURSE.items()})

    return df


def add_blood_features(df, base_dir=None, use_train_snapshot=False):
    """
    sire_stats.csv（種牡馬成績）と horse_master.csv（各馬の父・母父）から
    血統特徴量を付与する。血統データは途中でもよい（無い馬はNaN）。

    use_train_snapshot=True のとき sire_stats_father_train.csv（≤2024集計）を使う。
      学習・バックテスト時はこちら → テスト年(2025)の結果が血統統計に漏れない。
    False（本番予測）のときは全期間版 sire_stats_father.csv を使う。
      _train版が存在しない場合は自動で全期間版にフォールバック。

    付与する特徴量:
      父系_今回距離適性  : その馬の父系の、今回距離帯での勝率
      母父系_今回距離適性 : 母父系の、今回距離帯での勝率
      父系_長距離勝率    : 父系の長距離勝率（長距離の弱点対策）
      母父系_長距離勝率  : 母父系の長距離勝率
      父系_芝ダ適性      : 今回の芝/ダートでの父系勝率
      父系_複勝率        : 父系全体の複勝率（地力の指標）
    """
    import os as _os
    bd = base_dir or _os.path.dirname(_os.path.abspath(__file__))
    horse_path = _os.path.join(bd, "horse_master.csv")
    # 学習/BTは ≤2024版（リーク対策）、本番は全期間版。_train が無ければ全期間版に自動フォールバック。
    _suffix = "_train" if use_train_snapshot else ""
    father_path = _os.path.join(bd, f"sire_stats_father{_suffix}.csv")
    bms_path    = _os.path.join(bd, f"sire_stats_bms{_suffix}.csv")
    if use_train_snapshot and not _os.path.exists(father_path):
        print("  [血統] _train版が無いため全期間版にフォールバック（リーク注意）")
        father_path = _os.path.join(bd, "sire_stats_father.csv")
        bms_path    = _os.path.join(bd, "sire_stats_bms.csv")
    else:
        print(f"  [血統] スナップショット: {_os.path.basename(father_path)}")

    # データが無ければ全部NaNで返す（血統未取得でも動くように）
    blood_cols = ["父系_今回距離適性", "母父系_今回距離適性",
                  "父系_長距離勝率", "母父系_長距離勝率",
                  "父系_芝ダ適性", "父系_複勝率"]
    if not (_os.path.exists(horse_path) and _os.path.exists(father_path)):
        for c in blood_cols:
            df[c] = np.nan
        return df

    # 各馬の父・母父を結合
    horse = pd.read_csv(horse_path)
    horse = horse[horse["horse_id"].astype(str).str.replace(".0", "", regex=False).str.strip() != "horse_id"]
    horse["horse_id"] = horse["horse_id"].astype(str).str.replace(".0", "", regex=False).str.strip()
    if "horse_id" in df.columns:
        df["horse_id"] = df["horse_id"].astype(str).str.replace(".0", "", regex=False).str.strip()
        # horse_id 欠損の補完: 学習データの大半(約82%)で horse_id が欠損しているため、
        # horse_id が存在する行から「馬名→horse_id」対応表を作り、欠損行を馬名で補完する。
        # これにより血統紐づきが約17%→約67%に向上する。
        df.loc[df["horse_id"].isin(["nan", "", "None"]), "horse_id"] = np.nan
        if "馬名" in df.columns:
            has_id = df[df["horse_id"].notna()]
            name2id = (has_id.dropna(subset=["馬名"])
                              .drop_duplicates("馬名")
                              .set_index("馬名")["horse_id"].to_dict())
            fill = df["馬名"].map(name2id)
            df["horse_id"] = df["horse_id"].fillna(fill)
        df = df.merge(horse[["horse_id", "父馬", "母父馬"]], on="horse_id", how="left")
    else:
        for c in blood_cols:
            df[c] = np.nan
        return df

    # 種牡馬成績を読み込み
    father = pd.read_csv(father_path)
    bms    = pd.read_csv(bms_path)
    # 辞書化（高速参照）: {種牡馬名: {指標: 値}}
    father_d = father.set_index("父名").to_dict("index") if "父名" in father.columns else {}
    bms_d    = bms.set_index("母父名").to_dict("index") if "母父名" in bms.columns else {}

    def dist_band_col(d):
        if pd.isna(d):
            return None
        if d <= 1400:
            return "短距離"
        elif d <= 2000:
            return "中距離"
        else:
            return "長距離"

    # ── ベクトル化（旧実装は df.iterrows() で全行ループしており、予測時に履歴絞り込みが
    #   空振り→全32万行にフォールバックすると新馬戦などで数十分ハングしていた。
    #   .map / np.select で同一の値を高速算出する（正常行では旧実装と完全一致）。──
    def _cmap(d, col):
        return {k: v.get(col, np.nan) for k, v in d.items()}

    sire   = df["父馬"]   if "父馬"   in df.columns else pd.Series(np.nan, index=df.index)
    bmsire = df["母父馬"] if "母父馬" in df.columns else pd.Series(np.nan, index=df.index)
    dist   = pd.to_numeric(df["距離"], errors="coerce") if "距離" in df.columns else pd.Series(np.nan, index=df.index)
    is_turf = pd.to_numeric(df["is_turf"], errors="coerce") if "is_turf" in df.columns else pd.Series(np.nan, index=df.index)

    # 距離帯（短/中/長）。旧 dist_band_col と同じ境界。距離NaN→"" で band無効扱い。
    band = pd.Series(
        np.select([dist <= 1400, dist <= 2000, dist > 2000],
                  ["短距離", "中距離", "長距離"], default=""),
        index=df.index)
    band_valid = band != ""

    def _by_band(d, prefix):
        s = sire if prefix == "父" else bmsire
        short = s.map(_cmap(d, f"{prefix}_短距離勝率"))
        mid   = s.map(_cmap(d, f"{prefix}_中距離勝率"))
        lng   = s.map(_cmap(d, f"{prefix}_長距離勝率"))
        return pd.Series(np.select(
            [band == "短距離", band == "中距離", band == "長距離"],
            [short.to_numpy(), mid.to_numpy(), lng.to_numpy()], default=np.nan),
            index=df.index)

    # 父系（すべて「父馬既知 かつ band有効」のときのみ値を持つ＝旧 if fs and band と同義）
    f_turf_win = sire.map(_cmap(father_d, "父_芝勝率"))
    f_dirt_win = sire.map(_cmap(father_d, "父_ダート勝率"))
    df["父系_今回距離適性"] = _by_band(father_d, "父")
    df["父系_長距離勝率"]   = sire.map(_cmap(father_d, "父_長距離勝率")).where(band_valid)
    df["父系_複勝率"]       = sire.map(_cmap(father_d, "父_複勝率")).where(band_valid)
    df["父系_芝ダ適性"]     = pd.Series(
        np.where(is_turf == 1, f_turf_win, np.where(is_turf == 0, f_dirt_win, np.nan)),
        index=df.index).where(band_valid & is_turf.notna())

    # 母父系
    df["母父系_今回距離適性"] = _by_band(bms_d, "母父")
    df["母父系_長距離勝率"]   = bmsire.map(_cmap(bms_d, "母父_長距離勝率")).where(band_valid)

    n_blood = df["父系_今回距離適性"].notna().sum()
    print(f"  血統特徴量を付与: {n_blood}/{len(df)}行 ({n_blood/len(df)*100:.1f}%)")

    return df


def _dist_band(d):
    """距離を帯に変換（短/中/長）。コースバイアス集計のキー。"""
    d = pd.to_numeric(d, errors="coerce")
    if pd.isna(d):
        return None
    if d <= 1400:
        return "短"
    elif d <= 2000:
        return "中"
    return "長"


def build_course_bias(input_csv="race_data_clean.csv", out_csv="course_bias.csv",
                      year_max=2024, base_dir=None):
    """競馬場×距離帯×芝ダ ごとの脚質バイアス（先行/差し有利度）を集計してCSV保存する。
    リーク防止のため year_max（既定2024）以前のデータのみで集計する。
    予測時も同じCSVを参照するので学習・予測で完全一致する（sire_statsと同じ静的方式）。

    出力指標:
      コース好走相対4角 : 着順3以内馬の平均相対4角位置(0-1)。小=前有利、大=差し有利
      コース先行勝率   : 4角3番手以内だった馬の勝率
      コース差し複勝率 : 4角4番手以降だった馬の複勝率
    """
    import os as _os
    bd = base_dir or _os.path.dirname(_os.path.abspath(__file__))
    inp = input_csv if _os.path.isabs(input_csv) else _os.path.join(bd, input_csv)
    outp = out_csv if _os.path.isabs(out_csv) else _os.path.join(bd, out_csv)

    df = pd.read_csv(inp, low_memory=False)
    df["race_id"] = df["race_id"].astype(str)
    df["年"] = df["race_id"].str[:4].astype(int)
    df = df[df["年"] <= year_max].copy()
    df["競馬場cd"] = df["race_id"].str[4:6].astype(int)
    df["着順_num"] = pd.to_numeric(df["着順_num"], errors="coerce")
    df["距離"] = pd.to_numeric(df["距離"], errors="coerce")

    # is_turf（無ければ馬場種別から）
    if "is_turf" not in df.columns and "馬場種別" in df.columns:
        df["is_turf"] = df["馬場種別"].astype(str).str.contains("芝").astype(float)
    df["is_turf"] = pd.to_numeric(df.get("is_turf"), errors="coerce")

    df["_4c"] = df["通過"].apply(_extract_4corner) if "通過" in df.columns else np.nan
    df["_n"]  = df.groupby("race_id")["馬番"].transform("count")
    df["_rel4c"] = df["_4c"] / df["_n"].replace(0, np.nan)
    df["_senko"] = (df["_4c"] <= 3).astype(float)
    df["_win"]   = (df["着順_num"] == 1).astype(float)
    df["_t3"]    = (df["着順_num"] <= 3).astype(float)
    df["_band"]  = df["距離"].apply(_dist_band)

    rows = []
    for (jyo, band, turf), g in df.groupby(["競馬場cd", "_band", "is_turf"]):
        if len(g) < 30:      # サンプル不足のコースはスキップ（NaN扱い）
            continue
        good = g[g["_t3"] == 1]
        senko = g[g["_senko"] == 1]
        sashi = g[g["_senko"] == 0]
        rows.append({
            "競馬場cd": jyo, "_band": band, "is_turf": turf,
            "コース好走相対4角": good["_rel4c"].mean(),
            "コース先行勝率": senko["_win"].mean() if len(senko) > 0 else np.nan,
            "コース差し複勝率": sashi["_t3"].mean() if len(sashi) > 0 else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(outp, index=False, encoding="utf-8-sig")
    print(f"  コースバイアステーブル保存 → {outp}（{len(out)}コース）")
    return out


def add_course_bias_features(df, base_dir=None):
    """コース脚質バイアス（実装1）とコース替わり（実装2）の特徴量を付与する。
    add_extra_advanced_features の後に呼ぶこと（回り_num/直線長_m/坂あり/先行馬フラグが前提）。"""
    import os as _os
    bd = base_dir or _os.path.dirname(_os.path.abspath(__file__))
    cb_path = _os.path.join(bd, "course_bias.csv")

    # ── 競馬場cd / 距離帯 / is_turf を用意 ──
    if "競馬場cd" not in df.columns:
        df["競馬場cd"] = df["race_id"].astype(str).str[4:6].astype(int)
    df["_band"] = df["距離"].apply(_dist_band)
    turf = pd.to_numeric(df.get("is_turf"), errors="coerce")

    # ── 実装1-A: コース全体のバイアステーブルをマージ ──
    bias_cols = ["コース好走相対4角", "コース先行勝率", "コース差し複勝率"]
    if _os.path.exists(cb_path):
        cb = pd.read_csv(cb_path)
        cb["is_turf"] = pd.to_numeric(cb["is_turf"], errors="coerce")
        df["is_turf"] = turf
        df = df.merge(cb, on=["競馬場cd", "_band", "is_turf"], how="left")
    else:
        for c in bias_cols:
            df[c] = np.nan

    # ── 実装1-B: 馬の脚質 × コース相性 ──
    # 馬の相対脚質: 過去平均先行指数(1角平均位置) / 出走頭数（小=先行馬）
    n_run = pd.to_numeric(df.get("出走頭数"), errors="coerce").replace(0, np.nan)
    rel_kyakushitsu = pd.to_numeric(df.get("過去平均先行指数"), errors="coerce") / n_run
    senko_flag = pd.to_numeric(df.get("先行馬フラグ"), errors="coerce")
    course_front = pd.to_numeric(df["コース好走相対4角"], errors="coerce")  # 小=前有利
    # 脚質コース適合: 馬の相対位置とコース好走位置が近いほど相性good（差の絶対値の負）
    df["脚質コース適合"] = -(rel_kyakushitsu - course_front).abs()
    # 先行有利コース×先行馬: コースが前有利(値小→1-値大)で先行馬なら加点
    df["先行有利コース×先行馬"] = senko_flag * (1.0 - course_front)
    # 差し有利コース×差し馬: コースが差し有利(値大)で差し馬なら加点
    df["差し有利コース×差し馬"] = (1.0 - senko_flag) * course_front

    # ── 実装2: コース替わり（前走→今回のコース形態変化）──
    df = sort_by_horse_time(df)
    g = df.groupby("馬名")
    for col, prev in [("回り_num", "前走回り_num"), ("直線長_m", "前走直線長_m"), ("坂あり", "前走坂あり")]:
        if col in df.columns:
            df[prev] = g[col].transform(lambda x: x.shift(1))
        else:
            df[prev] = np.nan
    # 回り替わり（右⇄左の方向転換）
    if "回り_num" in df.columns:
        df["回り替わりフラグ"] = ((df["前走回り_num"].notna()) &
                                 (df["前走回り_num"] != pd.to_numeric(df["回り_num"], errors="coerce"))).astype(float)
        df.loc[df["前走回り_num"].isna(), "回り替わりフラグ"] = np.nan
    else:
        df["回り替わりフラグ"] = np.nan
    # 直線長変化（正=長い直線へ＝差し有利化）
    if "直線長_m" in df.columns:
        df["直線長変化"] = pd.to_numeric(df["直線長_m"], errors="coerce") - df["前走直線長_m"]
    else:
        df["直線長変化"] = np.nan
    # 坂替わり（平坦⇄急坂）
    if "坂あり" in df.columns:
        df["坂替わりフラグ"] = ((df["前走坂あり"].notna()) &
                              (df["前走坂あり"] != pd.to_numeric(df["坂あり"], errors="coerce"))).astype(float)
        df.loc[df["前走坂あり"].isna(), "坂替わりフラグ"] = np.nan
    else:
        df["坂替わりフラグ"] = np.nan
    # 初コースフラグ（その馬がその競馬場cd初出走なら1）
    # df は既に (馬名, race_id) でソート済み。馬ごとに競馬場の初出現を判定する。
    def _first_course(sub):
        seen = set(); out = []
        for cd in sub:
            out.append(1.0 if cd not in seen else 0.0)
            seen.add(cd)
        return out
    df["初コースフラグ"] = np.concatenate(
        [_first_course(s["競馬場cd"].tolist()) for _, s in df.groupby("馬名", sort=True)]
    )

    df = df.drop(columns=["_band"], errors="ignore")
    return df


def add_interaction_features(df):
    """
    交互作用特徴量を追加（高精度化）：
    ① 距離×馬場状態 過去成績
    ② 距離×クラス 過去成績
    ③ 馬場状態×脚質 相性
    ④ 前走着順×人気 乖離（前走好走なのに人気薄）
    ⑤ 斤量×年齢 負担指数
    """
    df = sort_by_horse_time(df)
    df["_win"] = (df["着順_num"] == 1).astype(float)

    # ── ① 距離カテゴリ×馬場状態 過去成績 ──────────────────────────
    grp_dist_baba = df.groupby(["馬名", "距離カテゴリ", "馬場状態_num"])
    df["距離×馬場_過去勝率"] = grp_dist_baba["_win"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["距離×馬場_過去平均着順"] = grp_dist_baba["着順_num"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # ── ② 距離カテゴリ×クラス 過去成績 ──────────────────────────────
    grp_dist_cls = df.groupby(["馬名", "距離カテゴリ", "クラス_num"])
    df["距離×クラス_過去勝率"] = grp_dist_cls["_win"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # ── ③ 芝ダート×脚質 相性（先行馬の芝ダート別成績） ─────────────
    if "先行指数" in df.columns:
        grp_turf_style = df.groupby(["馬名", "is_turf"])
        df["芝ダート×先行_過去勝率"] = grp_turf_style["_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    else:
        df["芝ダート×先行_過去勝率"] = np.nan

    # ── ④ 前走着順×人気 乖離スコア ────────────────────────────────
    # 前走好走（3着以内）なのに今回人気薄 → 高期待値候補
    if "前走着順" in df.columns and "人気" in df.columns:
        prev_good = (df["前走着順"] <= 3).astype(float)
        pop_low   = (df["人気"] >= 5).astype(float)
        df["前走好走×人気薄"] = prev_good * pop_low
        # 前走着順と人気の差（前走着順が良いのに人気が低い = プラス）
        df["前走着順×人気_乖離"] = df["人気"] - df["前走着順"].fillna(df["人気"])
    else:
        df["前走好走×人気薄"]   = np.nan
        df["前走着順×人気_乖離"] = np.nan

    # ── ⑤ 斤量×年齢 負担指数 ────────────────────────────────────────
    # 若い馬（2〜3歳）は斤量増に弱い
    df["斤量×年齢_負担"] = df["斤量"] * (1 + 1 / df["年齢"].clip(lower=2))

    # ── ⑥ 距離変化×前走着順 相性 ─────────────────────────────────────
    # 距離延長で前走好走の馬は好走しやすい
    if "距離変化" in df.columns and "前走着順" in df.columns:
        df["距離延長×前走好走"] = (
            (df["距離変化"] > 0).astype(float) *
            (df["前走着順"] <= 3).fillna(0).astype(float)
        )
        df["距離短縮×前走好走"] = (
            (df["距離変化"] < 0).astype(float) *
            (df["前走着順"] <= 3).fillna(0).astype(float)
        )
    else:
        df["距離延長×前走好走"] = np.nan
        df["距離短縮×前走好走"] = np.nan

    # ── ⑦ 前走間隔×距離変化 交互作用 ──────────────────────────────────
    # 長期休養明けの距離延長は特にリスクが高い（体力・ペース感覚両面で不安）
    if "前走間隔" in df.columns and "距離変化" in df.columns:
        _kyukan = df["前走間隔"].fillna(0)
        _dchg   = df["距離変化"].fillna(0)
        df["休養明け×距離延長"] = ((_kyukan >= 12).astype(float)) * (_dchg.clip(lower=0) / 400.0)
        df["連闘×距離短縮"]     = ((_kyukan <= 1).astype(float)) * ((-_dchg).clip(lower=0) / 400.0)
    else:
        df["休養明け×距離延長"] = np.nan
        df["連闘×距離短縮"]     = np.nan

    # ── ⑧ 前走着差×前走人気 乖離（接戦2着なのに今回人気薄 = 狙い目） ──
    if "前走着差_秒" in df.columns and "人気" in df.columns:
        # 前走が接戦（着差_秒<=0.3）かつ今回人気薄（人気>=5）= 見逃し候補
        _close_finish = (df["前走着差_秒"].fillna(99) <= 0.3).astype(float)
        _pop_low      = (pd.to_numeric(df["人気"], errors="coerce") >= 5).fillna(0).astype(float)
        df["接戦×人気薄"] = _close_finish * _pop_low
    else:
        df["接戦×人気薄"] = np.nan

    df = df.drop(columns=["_win"], errors="ignore")
    return df


def build_features(csv_path="race_data_clean.csv", out_path="race_features.csv", year_max=2024):
    print("データ読み込み中...")
    print(f"コースバイアステーブルを集計中（{year_max}以前）...")
    build_course_bias(csv_path, "course_bias.csv", year_max=year_max)
    df = load_and_prepare(csv_path)
    print("特徴量を生成中（過去成績・騎手・血統・コース・交互作用、時間がかかります）...")
    df = _run_feature_pipeline(df, use_train_snapshot=True)  # 学習: 血統は≤2024版でリーク防止

    FEATURE_COLS = [
        "race_id", "馬名", "着順_num",
        "単勝オッズ",
        "枠番", "馬番", "斤量", "斤量_相対",
        "年齢", "is_male", "is_female", "is_castrated",
        "馬体重", "体重増減", "馬体重_相対",
        "人気",
        "上り", "出走頭数", "競馬場cd", "レース番号", "日", "回",
        "過去出走数", "過去平均着順", "過去勝率", "過去複勝率",
        "過去平均上り", "直近3走平均着順",
        "過去平均タイム秒", "直近3走平均タイム秒", "過去最速タイム秒",
        "直近3走平均上り", "過去平均体重増減",
        "体重増減_過去標準偏差", "体重増減_異常度",
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
        # コース形態（B-2）
        "回り_num", "回り別_過去勝率", "回り別_過去複勝率",
        "直線長_m", "坂あり",
        # 脚質
        "過去平均先行指数", "先行馬フラグ",
        "想定先行馬数", "想定先行馬率", "他馬想定先行馬数", "差し馬×ハイペース想定",
        # 精密展開特徴（2026-07-27・通過順位ベース連続値）
        # ※ここに書き漏らすと計算されてもCSV保存時に落ちる（7/27に実際に発生）
        "脚質スコア", "前走相対位置", "想定逃げ馬数", "先行圧",
        "差し×先行圧", "単騎逃げ", "逃げ争い", "コース脚質バイアス",
        # 開催時期
        "開催月", "開催季節",
        # 前走情報
        "前走着順", "前走上り", "前走距離", "距離変化", "クラス変化",
        "前走着差_秒", "過去平均着差_秒", "近5走平均着差_秒", "近5走着差_std",
        "前走4角位置", "過去平均4角位置", "前走脚質指数",
        "芝ダート変更",
        "重賞出走フラグ", "過去重賞出走数",
        # 連続好走・成長
        "連続複勝フラグ", "連続勝利フラグ", "近走改善度",
        # タイム差
        "平均タイム差",
        # 騎手×競馬場・乗り替わり・斤量変化・前走条件変化
        "騎手競馬場勝率", "乗り替わり", "斤量変化", "連闘", "休み明け", "負担率",
        # 交互作用特徴量
        "距離×馬場_過去勝率", "距離×馬場_過去平均着順",
        "距離×クラス_過去勝率",
        "芝ダート×先行_過去勝率",
        "前走好走×人気薄", "前走着順×人気_乖離",
        "斤量×年齢_負担",
        "距離延長×前走好走", "距離短縮×前走好走",
        "休養明け×距離延長", "連闘×距離短縮",
        "接戦×人気薄",
        # 血統特徴量
        "父系_今回距離適性", "母父系_今回距離適性",
        "父系_長距離勝率", "母父系_長距離勝率",
        "父系_芝ダ適性", "父系_複勝率",
        # レース内ランク
        "レース内_過去勝率ランク", "レース内_直近3走平均着順ランク",
        "レース内_過去平均上りランク", "レース内_騎手勝率ランク",
        # 距離変化系
        "距離変化比率", "大幅延長フラグ", "大幅短縮フラグ", "距離延長幅",
        "長距離フラグ", "大幅延長×長距離", "距離延長×先行",
        # 距離適性・スタミナ系
        "経験最長距離", "経験範囲超過", "延長×距離経験不足", "長距離複勝率",
        "前走余力", "延長×前走余力",
        # 騎手直近トレンド
        "騎手直近勝率", "騎手直近複勝率", "騎手調子トレンド",
        # 直近5走統計
        "直近5走勝利数", "直近5走複勝数", "直近5走平均着順",
        # 着順安定性（複勝精度向け）
        "過去着順_std", "直近5走着順_std", "直近5走着外率",
        # ターゲットエンコーディング（B1: 馬主・スムージング版）
        "騎手勝率_sm", "調教師勝率_sm",
        "馬主勝率", "馬主複勝率", "馬主勝率_sm",
        # コース脚質バイアス（実装1）
        "コース好走相対4角", "コース先行勝率", "コース差し複勝率",
        "脚質コース適合", "先行有利コース×先行馬", "差し有利コース×差し馬",
        # コース替わり（実装2）
        "回り替わりフラグ", "直線長変化", "坂替わりフラグ", "初コースフラグ",
        # レース内偏差値化（相対競争の強調）
        "過去勝率_レース内偏差", "過去複勝率_レース内偏差", "直近3走平均着順_レース内偏差",
        "過去平均上り_レース内偏差", "過去獲得賞金累計_レース内偏差", "騎手勝率_レース内偏差",
        "父系_今回距離適性_レース内偏差", "直近5走平均着順_レース内偏差",
        # メンバーレベル（相手の強さ）
        "メンバー過去勝率平均", "メンバー過去勝率最大", "過去勝率_対相手",
        "メンバークラス平均", "メンバー賞金平均", "賞金_対相手",
        # 賞金累計
        "過去獲得賞金累計", "過去平均獲得賞金",
    ]

    # ※乗り替わり・斤量変化・連闘・休み明け・負担率は _run_feature_pipeline 内の
    #   add_rotation_weight_features() へ移動した（2026-07-31）。
    #   ここに置いていたため学習でしか作られず、予測経路では欠落していた。
    #   その結果これらの相対化列も作られず、モデルが要求する297列に対し
    #   予測時は272列しか揃わずLightGBMが落ちていた。

    # 相対化列(_R偏差/_R順)は add_all_relative_features が動的に作るので、
    # FEATURE_COLS に手書きせず、ここで自動的に保存対象へ加える。
    # ※過去に「計算したのに保存ホワイトリストへ書き忘れてCSVから落ちる」事故があった。
    rel = [c for c in df.columns if c.endswith(("_R偏差", "_R順"))]
    # 速度/上り指数の履歴も同様に自動で保存対象へ入れる（2026-08-03追加）。
    # ここに書き忘れると計算だけしてCSVから落ちる（過去に同じ事故あり）。
    spd = [c for c in SPEED_HIST_COLS if c in df.columns]
    chk = [c for c in CHOKYO_COLS if c in df.columns]     # 調教（2026-08-06追加）
    keep = [c for c in FEATURE_COLS if c in df.columns] + \
           [c for c in rel + spd + chk if c not in FEATURE_COLS]
    out = df[keep]
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"保存完了 → {out_path}（{len(out)}行 × {len(out.columns)}列"
          f"／うち相対化 {len(rel)}列）")
    return out


def add_distance_stamina_features(df):
    """距離適性・スタミナ系特徴量（経験最長距離・長距離複勝率・前走余力 など）。
    add_horse_history_features() 後に呼ぶこと（前走着順・距離延長幅が前提）。"""
    df = sort_by_horse_time(df)

    _cur_dist = pd.to_numeric(df["距離"], errors="coerce")

    # 経験最長距離（過去にこなした最長距離、shift(1)でリーク防止）
    # ※ groupbyは列アクセスのたびに再生成し pandas バージョン差異を回避
    df["経験最長距離"] = df.groupby("馬名")["距離"].transform(
        lambda x: pd.to_numeric(x, errors="coerce").shift(1).expanding().max()
    )

    # 経験範囲超過（今回距離が経験最長を超える量、正値=未知領域）
    df["経験範囲超過"] = (_cur_dist - df["経験最長距離"]).clip(lower=0)

    # 延長×距離経験不足（延長幅 × 未経験フラグ）
    _encho = df["距離延長幅"].values if "距離延長幅" in df.columns else np.zeros(len(df))
    _mikeiken = (_cur_dist > df["経験最長距離"]).astype(float).values
    df["延長×距離経験不足"] = _encho * _mikeiken

    # 長距離複勝率（2001m以上での過去top3率、shift(1)でリーク防止）
    _is_long   = (pd.to_numeric(df["距離"], errors="coerce") >= 2001).astype(float)
    _top3_flag = (df["着順_num"] <= 3).fillna(False).astype(float)
    _long_top3 = (_is_long * _top3_flag)

    df["_is_long_tmp"]   = _is_long
    df["_long_top3_tmp"] = _long_top3
    _long_cnt = df.groupby("馬名")["_is_long_tmp"].transform(
        lambda x: x.shift(1).expanding().sum()
    )
    _long_t3 = df.groupby("馬名")["_long_top3_tmp"].transform(
        lambda x: x.shift(1).expanding().sum()
    )
    df["長距離複勝率"] = np.where(_long_cnt > 0, _long_t3 / _long_cnt, np.nan)
    df = df.drop(columns=["_is_long_tmp", "_long_top3_tmp"], errors="ignore")

    # 前走余力（1着=1.0、2-3着=0.5、それ以外=0.0）
    _prev_chaku = df["前走着順"].values if "前走着順" in df.columns else np.full(len(df), np.nan)
    df["前走余力"] = np.where(
        pd.isna(_prev_chaku), np.nan,
        np.where(_prev_chaku == 1, 1.0,
        np.where(_prev_chaku <= 3, 0.5, 0.0))
    )

    # 延長×前走余力（距離延長が大きいほど前走余力が重要）
    df["延長×前走余力"] = _encho * df["前走余力"].values

    return df


def add_field_relative_features(df):
    """レース内偏差値化（相対競争の強調）とメンバーレベル（相手の強さ）を付与する。
    全特徴量が揃った後（pipeline末尾）に呼ぶこと。競馬は相対競争なので、
    絶対値より『そのレースメンバー内での相対位置』が効く。"""
    g = df.groupby("race_id")

    # ── レース内偏差値（zスコア）: 主要な連続特徴量をレース内で標準化 ──
    dev_cols = [
        "過去勝率", "過去複勝率", "直近3走平均着順", "過去平均上り",
        "過去獲得賞金累計", "騎手勝率", "父系_今回距離適性", "直近5走平均着順",
    ]
    for c in dev_cols:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            m = g[c].transform("mean")
            s = g[c].transform("std").replace(0, np.nan)
            df[f"{c}_レース内偏差"] = (x - m) / s

    # ── メンバーレベル（相手の強さ）──
    if "過去勝率" in df.columns:
        df["メンバー過去勝率平均"] = g["過去勝率"].transform("mean")
        df["メンバー過去勝率最大"] = g["過去勝率"].transform("max")
        df["過去勝率_対相手"] = pd.to_numeric(df["過去勝率"], errors="coerce") - df["メンバー過去勝率平均"]
    if "クラス_num" in df.columns:
        df["メンバークラス平均"] = g["クラス_num"].transform("mean")
    if "過去獲得賞金累計" in df.columns:
        df["メンバー賞金平均"] = g["過去獲得賞金累計"].transform("mean")
        df["賞金_対相手"] = (pd.to_numeric(df["過去獲得賞金累計"], errors="coerce")
                            - df["メンバー賞金平均"])
    return df


# 相対化から除く列（識別子・結果・市場評価・既に相対化済みのもの）
#
# ★当日結果そのものの列を必ず入れること（2026-07-31の事故）★
#   タイム秒/賞金/_1角位置/_4角位置/_chakusa_sec/先行指数 は「そのレースの結果」で、
#   学習データには存在するが予測時には存在しない。これを相対化して学習に混ぜたため、
#   予測時に該当12列が全欠損となりMF複勝率が0.182〜0.185の横並びに潰れ、
#   印が人気と大きく乖離した（BT: ◎平均1.9番人気 → 実際8〜13番人気）。
#   「前走〜」「過去〜」は履歴なので予測時にも存在し、除外してはいけない。
# ── レース内で全馬が同じ値になる列（2026-08-03に判明）────────────────────
# これらを add_all_relative_features に通すと:
#   _R偏差 … レース内の標準偏差が0なので全行NaN（実測で欠損100%）
#   _R順   … 全馬同値を method="first" で順位付けするので意味のない連番
# となり、FEATURE_COLS_MF の中に「中身のない列」が44本（22×2形態）できていた。
# 一方、生の値そのものは有用で、コース系4列を素のまま足すと +0.7pt（raw_ablation.py）。
# ※相対化を導入した2026-07-31の設計漏れ。REL_BASE_COLS からこれらを外し、
#   代わりに生の列を FEATURE_COLS_MF へ入れるのが正しい形。
RACE_CONST_COLS = [
    "クラス_num", "コース先行勝率", "コース好走相対4角", "コース差し複勝率",
    "メンバークラス平均", "メンバー賞金平均", "メンバー過去勝率平均",
    "メンバー過去勝率最大", "レース番号", "先行圧", "出走頭数", "回り_num",
    "想定先行馬数", "想定先行馬率", "想定逃げ馬数", "日", "直線長_m",
    "競馬場cd", "距離", "距離カテゴリ", "開催季節", "開催月",
]

# 生のまま特徴量に採用すべき列（raw_ablation.py の実測で採否を決めた）。
#   コース系  +0.7pt → 採用
#   メンバー系 ±0.0pt → 見送り
#   展開系    -0.4pt → 見送り
#   血統/馬主 -1.2pt → 見送り（集計方法に問題がある可能性。要調査）
RAW_ADOPT_COLS = [
    "コース脚質バイアス", "コース先行勝率", "コース差し複勝率", "コース好走相対4角",
]

_REL_SKIP_EXACT = {
    "race_id", "馬名", "馬番", "枠番", "着順_num", "上り", "人気", "単勝オッズ",
    "出走頭数", "レース番号", "競馬場cd", "日", "回", "距離", "クラス_num",
    "馬場状態_num", "is_turf", "回り_num", "開催月", "開催季節",
    # ↓当日結果に由来する列（予測時は未確定）
    "タイム秒", "賞金", "_1角位置", "_2角位置", "_3角位置", "_4角位置",
    "_chakusa_sec", "先行指数", "着差", "通過",
}
_REL_SKIP_SUB = ("レース内", "_R偏差", "_R順", "対相手", "メンバー", "_std")


# 相対化の対象列（固定）。学習と予測で必ず同じ集合になるようリスト化する。
# 2026-07-31: 動的選択にしていたため学習297列/予測272列とズレ、予測が落ちた。
# 変更するときは MFモデルの再学習が必要（use_cols と一致していないと動かない）。
REL_BASE_COLS = [
    "クラス_num", "クラス変化", "コース先行勝率",
    "コース好走相対4角", "コース差し複勝率", "コース脚質バイアス",
    "メンバークラス平均", "メンバー賞金平均", "メンバー過去勝率平均",
    "メンバー過去勝率最大", "レース番号", "他馬想定先行馬数",
    "体重増減", "体重増減_異常度", "先行圧",
    "先行有利コース×先行馬", "出走頭数", "前走4角位置",
    "前走上り", "前走余力", "前走着差_秒",
    "前走着順", "前走脚質指数", "前走距離",
    "前走間隔", "同距離過去勝率", "同距離過去平均着順",
    "回り_num", "回り別_過去勝率", "回り別_過去複勝率",
    "差し×先行圧", "差し有利コース×差し馬", "差し馬×ハイペース想定",
    "年齢", "延長×前走余力", "延長×距離経験不足",
    "想定先行馬数", "想定先行馬率", "想定逃げ馬数",
    "斤量", "斤量_相対", "斤量×年齢_負担",
    "斤量変化", "日", "枠番",
    "父系_今回距離適性", "父系_勝率", "父系_芝ダ適性",
    "父系_複勝率", "父系_長距離勝率", "直線長_m",
    "直線長変化", "直近3走平均タイム秒", "直近3走平均上り",
    "直近3走平均着順", "直近5走勝利数", "直近5走平均着順",
    "直近5走着外率", "直近5走着順_std", "直近5走複勝数",
    "競馬場cd", "競馬場過去勝率", "競馬場過去平均着順",
    "経験最長距離", "経験範囲超過", "脚質コース適合",
    "脚質スコア", "芝ダート×先行_過去勝率", "調教師勝率",
    "調教師勝率_sm", "調教師複勝率", "負担率",
    "距離", "距離×クラス_過去勝率", "距離×馬場_過去平均着順",
    "距離カテゴリ", "距離別過去平均上り", "距離別過去平均着順",
    "距離変化", "距離変化比率", "距離延長×先行",
    "距離延長幅", "近5走平均着差_秒", "近5走着差_std",
    "近走改善度", "逃げ争い", "速度_伸び",
    "速度_前走", "速度_直近3", "速度_直近3_順",
    "速度_過去平均", "速度_過去平均_順", "速度_過去最高",
    "速度_過去最高_順", "過去出走数", "過去勝率",
    "過去平均4角位置", "過去平均タイム秒", "過去平均上り",
    "過去平均体重増減", "過去平均先行指数", "過去平均獲得賞金",
    "過去平均着差_秒", "過去平均着順", "過去最速タイム秒",
    "過去最速上り", "過去獲得賞金累計", "過去着順_std",
    "過去複勝率", "過去重賞出走数", "開催季節",
    "開催月", "馬主勝率", "馬主勝率_sm",
    "馬主複勝率", "馬体重", "馬体重_相対",
    "騎手勝率", "騎手勝率_sm", "騎手直近複勝率",
    "騎手競馬場勝率", "騎手複勝率",
]

# ── 差し替え用の新しい素材リスト（2026-08-03準備・まだ未使用）──────────────
# レース内定数を除いたもの。122列 → 100列。
# 切り替えは「REL_BASE_COLS の再生成」と「MFモデルの再学習」を必ずセットで行う。
# 片方だけ変えると、モデルが要求する列とCSVの列が食い違って予測が落ちる
# （2026-07-31に同じ形で本番が停止した）。手順は apply_v2.py にまとめてある。
REL_BASE_COLS_V2 = [c for c in REL_BASE_COLS if c not in RACE_CONST_COLS]

# 切り替えはフラグファイルの有無で行う（環境変数だとタスクスケジューラから
# 起動されるプロセスに引き継がれず、学習と予測で食い違うため）。
# 作成/削除は apply_v2.py が行う。手で作らないこと。
V2_FLAG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "features_v2.flag")
if os.path.exists(V2_FLAG):
    REL_BASE_COLS = REL_BASE_COLS_V2


# ── 調教（坂路）特徴量（2026-08-06追加）────────────────────────────────
# 2026-08-04に一度見送った経緯がある。当時は元データ(chokyo_hc.csv)が無く、
# 8/2の予想37レースで調教データ0件＝「学習時は値があるのに予測時は必ず欠損」
# という不一致を作るところだった。
# 8/6にJV-LinkのSLOP(坂路)がセットアップモードで取得できると分かったので、
# 供給経路ができた前提で組み込む。
#
# ⚠️予測時にも値が入ることが必須。そのため chokyo_features.py（過去分の一括生成・
#   race_data_clean.csv とRAレコードに依存）は使わず、chokyo_hc.csv から
#   その場で計算する。レース日より前の調教だけを見るのでリークは無い。
CHOKYO_COLS = ["chk_last4f", "chk_last1f", "chk_days", "chk_n14", "chk_best4f"]
_CHOKYO_CACHE = None


def _load_chokyo(base_dir=None):
    """chokyo_hc.csv を馬ごとの配列にして返す。無ければNone。"""
    global _CHOKYO_CACHE
    if _CHOKYO_CACHE is not None:
        return _CHOKYO_CACHE
    base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "chokyo_hc.csv")
    if not os.path.exists(path):
        print("  調教データなし（chokyo_hc.csv）→ 調教特徴量はスキップ")
        _CHOKYO_CACHE = {}
        return _CHOKYO_CACHE
    try:
        ck = pd.read_csv(path, dtype={"horse_id": str, "調教日": str})
        ck["date"] = pd.to_datetime(ck["調教日"], format="%Y%m%d", errors="coerce")
        ck = ck.dropna(subset=["date", "time4f"]).sort_values(["horse_id", "date"])
        _CHOKYO_CACHE = {h: (g["date"].values, g["time4f"].values, g["lap1f"].values)
                         for h, g in ck.groupby("horse_id")}
        print(f"  調教データ読込: {len(ck):,}行 / {len(_CHOKYO_CACHE):,}頭")
    except Exception as e:
        print(f"  調教データの読込に失敗（スキップ）: {e}")
        _CHOKYO_CACHE = {}
    return _CHOKYO_CACHE


def add_chokyo_features(df, base_dir=None):
    """レース日より前の坂路調教から特徴量を作る。学習・予測の両方で同じ計算をする。"""
    hd = _load_chokyo(base_dir)
    for c in CHOKYO_COLS:
        df[c] = np.nan
    if not hd or "horse_id" not in df.columns:
        return df
    df = attach_race_date(df)
    if "_race_dt" not in df.columns:
        return df
    hid = df["horse_id"].astype(str).values
    rdt = pd.to_datetime(df["_race_dt"], errors="coerce").values
    out = np.full((len(df), 5), np.nan)
    win14 = np.timedelta64(14, "D")
    for i in range(len(df)):
        h = hd.get(hid[i])
        if h is None or pd.isna(rdt[i]):
            continue
        dates, t4, l1 = h
        idx = np.searchsorted(dates, rdt[i])   # レース日より前の調教まで
        if idx == 0:
            continue
        last = idx - 1
        days = (rdt[i] - dates[last]) / np.timedelta64(1, "D")
        if days <= 0:                          # 当日の調教は使わない
            last -= 1
            if last < 0:
                continue
            days = (rdt[i] - dates[last]) / np.timedelta64(1, "D")
        lo = np.searchsorted(dates, rdt[i] - win14)
        seg = t4[lo:idx]
        out[i] = (t4[last], l1[last], days, idx - lo,
                  np.nanmin(seg) if len(seg) else np.nan)
    for j, c in enumerate(CHOKYO_COLS):
        df[c] = out[:, j]
    return df


# ── 速度指数・上り指数（2026-08-03追加）──────────────────────────────────
# 実験用の final_features.py だけが持っていて本番に無かった特徴量。
# 同じ買い方でも実験モデルはワイド★軸127.6%、本番モデルは81.4%と46pt差があり、
# 印の質も 1位複勝率 62.8% 対 57.1% と開いていた。その主因がこの一群。
#
# ⚠️最重要の注意 ⚠️
#   速度指数・上り指数そのものは「当日のタイム・上り」から作る＝当日結果である。
#   絶対に特徴量にしてはいけない。2026-07-31に当日結果を相対化して3度目の
#   リークを出したのと同じ形。ここでは shift(1) を挟んだ過去分だけを残し、
#   生の指数は計算し終えたら必ず drop する。
SPEED_BASE_CSV = "speed_baseline.csv"
SPEED_HIST_COLS = [
    "速度_過去平均", "速度_過去最高", "速度_過去標準偏差", "速度_直近3",
    "速度_前走", "速度_伸び",
    "上指_過去平均", "上指_過去最高", "上指_直近3", "上指_前走",
    "速度_同距離帯", "速度_同芝ダ", "速度_同場", "速度_同馬場",
]


def _speed_baseline(df, base_dir=None):
    """(競馬場cd, 距離, is_turf, 馬場状態_num) ごとの基準タイムを返す。

    学習時に作って保存し、予測時は読み込む。毎回その場のデータから中央値を
    取ると学習と予測で基準がズレ、同じ馬が違う速度指数になってしまう
    （コース脚質バイアスで実際に起きた問題と同じ）。
    """
    base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, SPEED_BASE_CSV)
    keys = ["競馬場cd", "距離", "is_turf", "馬場状態_num"]
    if os.path.exists(path):
        try:
            b = pd.read_csv(path)
            if len(b) and set(keys).issubset(b.columns):
                return b
        except Exception as e:
            print(f"  速度基準の読込に失敗（作り直します）: {e}")
    d = df.copy()
    for k in keys:
        d[k] = pd.to_numeric(d[k], errors="coerce")
    b = (d.dropna(subset=keys + ["タイム秒"])
           .groupby(keys, observed=True)["タイム秒"].median()
           .rename("基準秒").reset_index())
    b = b[b["基準秒"] > 0]
    try:
        b.to_csv(path, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"  速度基準の保存に失敗（続行）: {e}")
    return b


def add_speed_index_features(df, base_dir=None):
    """速度指数・上り指数を作り、その馬の過去分だけを特徴量として残す。"""
    if "タイム秒" not in df.columns:
        for c in SPEED_HIST_COLS:
            df[c] = np.nan
        return df

    keys = ["競馬場cd", "距離", "is_turf", "馬場状態_num"]
    for k in keys:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    b = _speed_baseline(df, base_dir)
    df = df.merge(b, on=keys, how="left")
    # 基準が引けない条件（新設コース等）は距離だけの中央値で埋める
    if df["基準秒"].isna().any():
        fb = (b.groupby("距離", observed=True)["基準秒"].mean()
                .rename("_基準fb").reset_index())
        df = df.merge(fb, on="距離", how="left")
        df["基準秒"] = df["基準秒"].fillna(df["_基準fb"])
        df = df.drop(columns=["_基準fb"])

    _sec = pd.to_numeric(df["タイム秒"], errors="coerce")
    df["_速度指数"] = (df["基準秒"] - _sec) / df["基準秒"] * 1000
    df.loc[df["_速度指数"].abs() > 200, "_速度指数"] = np.nan

    _agari = pd.to_numeric(df.get("上り"), errors="coerce")
    if _agari is not None and _agari.notna().any():
        _ab = (df.dropna(subset=keys).assign(_a=_agari)
                 .groupby(keys, observed=True)["_a"].transform("median"))
        df["_上り指数"] = (_ab - _agari) / _ab * 1000
        df.loc[df["_上り指数"].abs() > 300, "_上り指数"] = np.nan
    else:
        df["_上り指数"] = np.nan

    # ここから先は必ず shift(1) を挟む＝自分のレースを見ない
    df = sort_by_horse_time(df)
    g = df.groupby("馬名", sort=False)
    df["速度_過去平均"] = g["_速度指数"].transform(lambda s: s.shift(1).expanding().mean())
    df["速度_過去最高"] = g["_速度指数"].transform(lambda s: s.shift(1).expanding().max())
    df["速度_過去標準偏差"] = g["_速度指数"].transform(lambda s: s.shift(1).expanding().std())
    df["速度_直近3"] = g["_速度指数"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["速度_前走"] = g["_速度指数"].transform(lambda s: s.shift(1))
    df["速度_伸び"] = df["速度_直近3"] - df["速度_過去平均"]
    df["上指_過去平均"] = g["_上り指数"].transform(lambda s: s.shift(1).expanding().mean())
    df["上指_過去最高"] = g["_上り指数"].transform(lambda s: s.shift(1).expanding().max())
    df["上指_直近3"] = g["_上り指数"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["上指_前走"] = g["_上り指数"].transform(lambda s: s.shift(1))

    # 条件別の過去平均。groupbyに必ず馬名を含める（含めないと他馬の行を跨ぎ、
    # 時系列順でないためリークになる）。
    df["_速度距離帯"] = (pd.to_numeric(df["距離"], errors="coerce") / 400).round()
    df["_速度道悪"] = (pd.to_numeric(df.get("馬場状態_num"), errors="coerce") >= 2).astype(float)
    for ks, nm in ((["馬名", "_速度距離帯"], "同距離帯"), (["馬名", "is_turf"], "同芝ダ"),
                   (["馬名", "競馬場cd"], "同場"), (["馬名", "_速度道悪"], "同馬場")):
        if not all(k in df.columns for k in ks):
            df[f"速度_{nm}"] = np.nan
            continue
        df[f"速度_{nm}"] = (df.groupby(ks, sort=False, observed=True)["_速度指数"]
                            .transform(lambda s: s.shift(1).expanding().mean()))

    # 生の指数と作業列は必ず捨てる（当日結果なので残すと事故る）
    df = df.drop(columns=[c for c in ("_速度指数", "_上り指数", "基準秒",
                                      "_速度距離帯", "_速度道悪") if c in df.columns])
    return df


def add_rotation_weight_features(df):
    """乗り替わり・斤量変化・連闘・休み明け・負担率を付ける。

    2026-07-31: もともと build_features（学習）の末尾にあり、予測経路の
    _run_feature_pipeline を通らなかった。そのため予測時にこれらと
    その相対化列が欠落し、モデルの要求列数(297)に届かず落ちていた。
    学習・予測の二重管理を無くすためパイプラインへ移設。
    """
    if "騎手" in df.columns or "斤量" in df.columns:
        df = sort_by_horse_time(df)
        if "騎手" in df.columns:
            prev_jockey = df.groupby("馬名")["騎手"].shift(1)
            df["乗り替わり"] = (prev_jockey.notna()
                              & (prev_jockey != df["騎手"])).astype(int)
        if "斤量" in df.columns:
            df["斤量変化"] = df.groupby("馬名")["斤量"].diff()
    if "前走間隔" in df.columns:
        df["連闘"] = (df["前走間隔"] <= 1).astype(float)
        df["休み明け"] = (df["前走間隔"] >= 12).astype(float)
        df.loc[df["前走間隔"].isna(), ["連闘", "休み明け"]] = float("nan")
    if "斤量" in df.columns and "馬体重" in df.columns:
        df["負担率"] = df["斤量"] / df["馬体重"].replace(0, float("nan"))
    return df


def add_all_relative_features(df, max_cols=90):
    """主要な数値特徴量をレース内で相対化する（偏差・順位の2形態）。

    2026-07-31追加。市場評価(人気・オッズ)を使わない予想では、絶対値より
    「そのレースの相手と比べてどうか」が圧倒的に効くと実測で判明したため。
      ・速度指数は生の値が重要度91位 → レース内偏差にすると4位
      ・血統も絶対値では埋もれるが、レース内順位にすると上位
      ・◎の複勝率 58.3% → 61.0%（+2.7pt・今回の改良で最大の効果）
    レース内だけで完結する計算なので、外部データもリークも無い。
    列数が増えすぎないよう、欠損が多い列とフラグ列は除き上限を設ける。
    """
    g = df.groupby("race_id")
    n = g["race_id"].transform("size")
    # 対象列は必ず REL_BASE_COLS（固定リスト）から選ぶ。
    # 2026-07-31: 当初は「欠損率と分散から動的に上位90列」を選んでいたが、
    #   学習(32万行)と予測(1レース9頭)で選ばれる列が変わり、
    #   モデルの要求列(297)に対し予測時は272列しか揃わずLightGBMが落ちた。
    #   列の集合は入力データに依存してはならないので固定リストにする。
    cand = [c for c in REL_BASE_COLS if c in df.columns
            and pd.api.types.is_numeric_dtype(df[c])]
    made = []
    for c in cand:
        x = pd.to_numeric(df[c], errors="coerce")
        gg = x.groupby(df["race_id"])
        # レース内で全馬が同じ値だと標準偏差0になり _R偏差 はNaNになる
        # （例: 逃げ争いが全馬0）。学習・予測とも同じ挙動なので不整合は無い。
        # ※0埋めにする改善は有効だが、モデルはNaN(→-999)で学習済みのため
        #   変更するときは必ず race_features 再生成とMF再学習をセットで行うこと。
        sd = gg.transform("std").replace(0, np.nan)
        df[f"{c}_R偏差"] = (x - gg.transform("mean")) / sd
        df[f"{c}_R順"] = gg.rank(ascending=False, method="first") / n
        made += [f"{c}_R偏差", f"{c}_R順"]
    df.attrs["relative_cols"] = made
    return df


def _run_feature_pipeline(df, use_train_snapshot=False):
    """load_and_prepare 済みの df に対し、学習と同じ特徴量関数を順に適用する。
    build_features（学習）と build_features_for_prediction（予測）で共通利用し、
    特徴量計算の二重管理（学習と予測のズレ）を防ぐ。
    use_train_snapshot: 血統統計に ≤2024版を使う（学習/BTのリーク対策）。"""
    df = add_horse_history_features(df)
    df = add_distance_stamina_features(df)
    df = add_race_relative_features(df)
    df = add_jockey_trainer_features(df)
    df["距離カテゴリ"] = pd.cut(
        df["レース番号"], bins=[0, 4, 8, 12], labels=[1, 2, 3]
    ).astype(float)
    df = sort_by_horse_time(df)
    dist_avg = []
    for _, group in df.groupby(["馬名", "距離カテゴリ"]):
        group = group.copy()
        group["距離別過去平均着順"] = group["着順_num"].shift(1).expanding().mean()
        dist_avg.append(group)
    if dist_avg:
        df = pd.concat(dist_avg).sort_values(["race_id", "馬番"]).reset_index(drop=True)
    df = add_extra_advanced_features(df)
    df = add_course_bias_features(df)
    df = add_blood_features(df, use_train_snapshot=use_train_snapshot)
    df = add_interaction_features(df)
    df = add_field_relative_features(df)
    df = add_speed_index_features(df)      # 2026-08-03: 速度/上り指数の過去履歴
    df = add_chokyo_features(df)           # 2026-08-06: 坂路調教（レース日より前のみ）
    df = add_rotation_weight_features(df)  # 2026-07-31: 学習専用だったものを移設
    df = add_all_relative_features(df)     # 2026-07-31: 全特徴のレース内相対化
    if "距離" in df.columns:
        df["距離"] = pd.to_numeric(df["距離"], errors="coerce")
        df["距離"] = df["距離"].fillna(df["距離"].median())
    return df


def build_features_for_prediction(race_df, history_df):
    """予測対象レース(race_df, 着順なし)に、履歴(history_df)から
    学習と同一の特徴量を付与する。学習(build_features)と同じ
    _run_feature_pipeline を通すので、距離適性・騎手トレンド・好調度・
    血統を含む全特徴量が、学習と完全に同じロジックで計算される。

    戻り値: 予測対象レースの行のみ（特徴量付き）。
    """
    race_df = race_df.copy()
    history_df = history_df.copy()
    race_df["race_id"] = race_df["race_id"].astype(str)
    history_df["race_id"] = history_df["race_id"].astype(str)
    target_ids = set(race_df["race_id"].unique())

    # 高速化: 全履歴(約32万行)ではなく、予測対象に関連するレースだけに絞る。
    # 各特徴量が必要とするデータ:
    #   ・馬の過去成績/距離適性/好調度 → 予測対象馬(馬名)の過去走
    #   ・騎手トレンド/騎手勝率 → 予測対象の騎手が乗った全レース
    #   ・調教師成績 → 予測対象の調教師のレース
    #   ・血統 → sire_stats CSV から引く(履歴不要)
    # これらの和集合だけ残せば、計算結果は全履歴使用時と同一になる。
    target_horses = target_jockeys = target_trainers = set()
    try:
        def _norm(s):
            return s.dropna().astype(str).str.strip()
        target_horses   = set(_norm(race_df["馬名"]))   if "馬名"   in race_df.columns else set()
        target_jockeys  = set(_norm(race_df["騎手"]))   if "騎手"   in race_df.columns else set()
        target_trainers = set(_norm(race_df["調教師"])) if "調教師" in race_df.columns else set()
        mask = pd.Series(False, index=history_df.index)
        if target_horses and "馬名" in history_df.columns:
            mask |= history_df["馬名"].astype(str).str.strip().isin(target_horses)
        if target_jockeys and "騎手" in history_df.columns:
            mask |= history_df["騎手"].astype(str).str.strip().isin(target_jockeys)
        if target_trainers and "調教師" in history_df.columns:
            mask |= history_df["調教師"].astype(str).str.strip().isin(target_trainers)
        if mask.any():
            history_df = history_df[mask].copy()
    except Exception as _e:
        print(f"  [予測] 履歴絞り込みエラー（安全弁で制限します）: {_e}")

    # ── 安全弁: 単一レース予測で全履歴(約32万行)を特徴量計算に回すと、新馬戦など
    #   絞り込みが空振りしたケースで数十分ハングする（add_horse_history_features 内の
    #   per-row ループが O(行数)・約11ms/行のため。将来ここをベクトル化予定）。
    #   過大なままなら「対象馬の過去走＋直近レース」に限定して上限を課す。
    #   通常レースは絞り込み後 約5〜11k行なので下記(20000)は no-op。新馬フォールバック時のみ
    #   発火し 32万行→2万行(約4分)に抑える。
    _MAX_PRED_HISTORY = 20000
    if len(history_df) > _MAX_PRED_HISTORY:
        keep = pd.Series(False, index=history_df.index)
        if target_horses and "馬名" in history_df.columns:
            keep |= history_df["馬名"].astype(str).str.strip().isin(target_horses)  # 対象馬走は必ず残す
        remain = _MAX_PRED_HISTORY - int(keep.sum())
        if remain > 0:
            recent_idx = history_df.loc[~keep].sort_values("race_id").tail(remain).index
            keep.loc[recent_idx] = True
        print(f"  [予測] 履歴が過大({len(history_df)}行) → {int(keep.sum())}行に制限（絞り込み空振りの安全弁）")
        history_df = history_df[keep].copy()

    # 予測対象に無い列を NaN で補い、履歴と列を揃える（縦結合のため）
    for col in history_df.columns:
        if col not in race_df.columns:
            race_df[col] = np.nan
    # 予測対象行の着順はまだ無い（未来）。集計の材料にならないよう NaN にする。
    for c in ["着順", "着順_num", "タイム", "上り", "通過", "着差", "賞金"]:
        if c in race_df.columns:
            race_df[c] = np.nan

    combined = pd.concat([history_df, race_df[history_df.columns]], ignore_index=True)
    combined = load_and_prepare(df=combined)
    combined = _run_feature_pipeline(combined, use_train_snapshot=False)  # 本番: 血統は全期間版

    # 予測対象レースの行だけ取り出す
    result = combined[combined["race_id"].isin(target_ids)].copy()
    return result


if __name__ == "__main__":
    df = build_features()
    print("\nサンプル:")
    print(df.head(3).to_string())