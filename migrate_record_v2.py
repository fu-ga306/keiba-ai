import os
import pandas as pd
import numpy as np

BASE_DIR    = r"c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai"
RECORD_FILE = os.path.join(BASE_DIR, "prediction_record_v2.csv")

REQUIRED_COLS = [
    "日付", "race_id", "jyo", "race",
    "honmei", "honmei_win_p", "taiko", "taiko_win_p",
    "ana", "ana_win_p",
    "honmei_odds", "honmei_ninki", "honmei_ev",
    "honmei_actual", "hit", "taiko_actual", "ana_actual",
]

if not os.path.exists(RECORD_FILE):
    print("prediction_record_v2.csv が見つかりません")
else:
    df = pd.read_csv(RECORD_FILE)
    print("既存カラム:", list(df.columns))

    added = []
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = np.nan
            added.append(col)

    # カラム順を統一
    df = df[REQUIRED_COLS]

    df.to_csv(RECORD_FILE, index=False, encoding="utf-8-sig")
    print("追加したカラム:", added)
    print("マイグレーション完了")
    print(df.head())
