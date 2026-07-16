# -*- coding: utf-8 -*-
"""JVのHR(払戻)レコードをパースして jv_payouts.csv を作る。
 レイアウト(1行622桁・実データから解読):
   102- 単勝×3   [馬番2+払戻9+人気2]
   141- 複勝×5   [馬番2+払戻9+人気2]
   206- 枠連×3   [枠2連結2+払戻9+人気2]
   245- 馬連×3   [馬番4+払戻9+人気3]
   293- ワイド×7 [馬番4+払戻9+人気3]
   405- 枠単×3   [常に空(JRA未発売)]
   453- 馬単×6   [馬番4+払戻9+人気3] ※着順順序つき
   549- 3連複×3  [馬番6+払戻9+人気3]
   603- 3連単×?  [馬番6+払戻9+人気4] ※着順順序つき・行末まで
 出力: race_id, 券種, 組み合わせ(ハイフン区切り・馬単/3連単は着順順), 払戻金, 人気
"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

LAYOUT = [
    ("単勝",   102, 3, 2, 2),
    ("複勝",   141, 5, 2, 2),
    ("枠連",   206, 3, 2, 2),
    ("馬連",   245, 3, 4, 3),
    ("ワイド", 293, 7, 4, 3),
    ("馬単",   453, 6, 4, 3),
    ("3連複",  549, 3, 6, 3),
]


def split_combo(c):
    return "-".join(c[i:i + 2] for i in range(0, len(c), 2))


def parse_line(t):
    rid = t[11:15] + t[19:27]
    rows = []
    for kind, start, n, clen, plen in LAYOUT:
        slot = clen + 9 + plen
        for k in range(n):
            s = start + k * slot
            combo, pay, pop = t[s:s + clen], t[s + clen:s + clen + 9], t[s + clen + 9:s + slot]
            if not combo.strip() or not combo.isdigit() or not pay.isdigit():
                continue
            if int(pay) == 0:
                continue
            if kind == "枠連":
                cb = "-".join(f"0{d}" for d in combo)   # '34'→'03-04'（枠は1桁連結）
            else:
                cb = split_combo(combo)
            rows.append({"race_id": rid, "券種": kind, "組み合わせ": cb,
                         "払戻金": int(pay), "人気": int(pop) if pop.isdigit() else None})
    # 3連単: 603から行末まで19桁チャンク
    s = 603
    while s + 19 <= len(t):
        combo, pay, pop = t[s:s + 6], t[s + 6:s + 15], t[s + 15:s + 19]
        if combo.isdigit() and pay.isdigit() and int(pay) > 0:
            rows.append({"race_id": rid, "券種": "3連単", "組み合わせ": split_combo(combo),
                         "払戻金": int(pay), "人気": int(pop) if pop.isdigit() else None})
        s += 19
    return rows


def main():
    rows = []
    seen = {}
    with open(os.path.join(BASE, "data", "jv", "RACE_HR.txt"), encoding="utf-8") as f:
        for line in f:
            if not line.startswith("HR"):
                continue
            t = line.rstrip("\n")
            rid = t[11:15] + t[19:27]
            # データ区分が新しいもの優先（同一race_idは後勝ち＝確定値）
            seen[rid] = t
    for rid, t in seen.items():
        rows.extend(parse_line(t))
    df = pd.DataFrame(rows)
    out = os.path.join(BASE, "jv_payouts.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    yr = df["race_id"].str[:4]
    print(f"保存: {out} ({len(df)}行 / {df['race_id'].nunique()}レース)")
    print("年別レース数:", yr.groupby(yr).apply(lambda s: df.loc[s.index, 'race_id'].nunique()).to_dict())

    # ── 検証: スクレイプ済み2025払戻と全件突合 ──
    sc = pd.read_csv(os.path.join(BASE, "payout_data.csv"), dtype={"組み合わせ": str})
    sc["race_id"] = sc["race_id"].astype(str)
    sc = sc[sc["race_id"].str.startswith("2025")]
    jv25 = df[df["race_id"].str.startswith("2025")]
    m = sc.merge(jv25, on=["race_id", "券種", "組み合わせ"], how="left", suffixes=("_sc", "_jv"))
    ok = (pd.to_numeric(m["払戻金_sc"]) == m["払戻金_jv"]).mean() * 100
    print(f"2025スクレイプ払戻との一致率: {ok:.2f}%  (突合{len(m)}行)")


if __name__ == "__main__":
    main()
