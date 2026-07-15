# -*- coding: utf-8 -*-
"""
当日の予想(today_predictions.csv)と実際のレース結果を比較し、
AI評価の精度を多角的に分析する。

分析内容:
  1. ◎(印)馬の着順分布 → 印がどれだけ当たっているか
  2. 予測順位 vs 実際着順の相関 → モデルの順位付け精度
  3. AI予測1位馬の勝率・連対率・複勝率(実績)
  4. 純粋AI評価(MF) vs 通常モデルの精度比較
  5. 戦略該当馬の実績
  6. 人気帯別の精度(人気を当てているだけか、妙味を出せているか)
出力はコンソール + accuracy_log.csv（日付別に蓄積）。
"""
import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
from result_tracker import get_race_result, BASE_DIR

PRED_FILE = os.path.join(BASE_DIR, "today_predictions.csv")
LOG_FILE  = os.path.join(BASE_DIR, "accuracy_log.csv")


def main():
    if not os.path.exists(PRED_FILE):
        print(f"予想ファイルがありません: {PRED_FILE}")
        return

    pred = pd.read_csv(PRED_FILE)
    pred["race_id"] = pred["race_id"].astype(str)
    race_ids = sorted(pred["race_id"].unique())
    print(f"対象レース数: {len(race_ids)}")

    # 各レースの結果を取得して予想とマージ
    merged_all = []
    for rid in race_ids:
        res = get_race_result(rid)
        if res is None:
            print(f"  結果取得失敗: {rid}")
            time.sleep(1)
            continue
        res = res[["馬名", "着順_num"]].copy()
        res["race_id"] = rid
        p = pred[pred["race_id"] == rid].copy()
        m = p.merge(res, on=["race_id", "馬名"], how="left")
        merged_all.append(m)
        print(f"  取得: {rid} ({len(m)}頭)")
        time.sleep(0.4)   # requests化で高速。過負荷防止の礼儀待機のみ

    if not merged_all:
        print("結果が取得できませんでした")
        return

    df = pd.concat(merged_all, ignore_index=True)
    df = df.dropna(subset=["着順_num"])
    print(f"\n結果と照合できた頭数: {len(df)}")

    # ── 1. 印(推奨ランク)別の着順分布 ──
    print("\n" + "=" * 55)
    print("1. 印別の実際の着順")
    print("=" * 55)
    if "推奨ランク" in df.columns:
        for mark in ["◎", "○", "▲", "△", "×"]:
            sub = df[df["推奨ランク"] == mark]
            if len(sub) == 0:
                continue
            win = (sub["着順_num"] == 1).mean() * 100
            ren = (sub["着順_num"] <= 2).mean() * 100
            fuku = (sub["着順_num"] <= 3).mean() * 100
            avg = sub["着順_num"].mean()
            print(f"  {mark}: 頭数{len(sub):3d}  勝率{win:5.1f}%  "
                  f"連対率{ren:5.1f}%  複勝率{fuku:5.1f}%  平均着順{avg:.1f}")
    # ◎妙（妙味軸=回収率エンジン）も照合
    if "妙味軸" in df.columns:
        sub = df[df["妙味軸"] == "◎妙"]
        if len(sub) > 0:
            win = (sub["着順_num"] == 1).mean() * 100
            fuku = (sub["着順_num"] <= 3).mean() * 100
            print(f"  ◎妙: 頭数{len(sub):3d}  勝率{win:5.1f}%  複勝率{fuku:5.1f}%  "
                  f"(BT: 勝率15.1%/単勝ROI167%)")

    # ── 2. 予測順位 vs 実着順の相関 ──
    print("\n" + "=" * 55)
    print("2. 予測順位 vs 実着順の精度")
    print("=" * 55)
    if "予測順位" in df.columns:
        d2 = df.dropna(subset=["予測順位"])
        corr = d2["予測順位"].corr(d2["着順_num"])
        print(f"  順位相関(スピアマン的): {corr:.3f}  (1に近いほど良い)")
        # 予測1位が実際に何着だったか
        top1 = d2[d2["予測順位"] == 1]
        if len(top1) > 0:
            print(f"  予測1位馬({len(top1)}レース): "
                  f"勝率{(top1['着順_num']==1).mean()*100:.1f}%  "
                  f"連対率{(top1['着順_num']<=2).mean()*100:.1f}%  "
                  f"複勝率{(top1['着順_num']<=3).mean()*100:.1f}%")

    # ── 3. 通常モデル vs 純粋AI(MF)の予測1位精度比較 ──
    print("\n" + "=" * 55)
    print("3. 通常モデル vs 純粋AI評価(MF) の1位精度")
    print("=" * 55)
    for col, label in [("勝ち確率", "通常モデル"), ("MF勝ち確率", "純粋AI(MF)")]:
        if col not in df.columns:
            continue
        # レースごとに該当列の最大値の馬を取得
        idx = df.groupby("race_id")[col].idxmax().dropna()
        tops = df.loc[idx]
        if len(tops) > 0:
            print(f"  {label}1位: "
                  f"勝率{(tops['着順_num']==1).mean()*100:.1f}%  "
                  f"連対率{(tops['着順_num']<=2).mean()*100:.1f}%  "
                  f"複勝率{(tops['着順_num']<=3).mean()*100:.1f}%")

    # ── 4. 戦略該当馬の実績 ──
    print("\n" + "=" * 55)
    print("4. 戦略該当馬の実績")
    print("=" * 55)
    if "該当戦略" in df.columns:
        strat = df[df["該当戦略"].notna() & (df["該当戦略"].astype(str) != "")
                   & (df["該当戦略"].astype(str) != "nan")]
        if len(strat) > 0:
            for s, g in strat.groupby("該当戦略"):
                win = (g["着順_num"] == 1).mean() * 100
                fuku = (g["着順_num"] <= 3).mean() * 100
                print(f"  {s}: 頭数{len(g):2d}  勝率{win:.1f}%  複勝率{fuku:.1f}%")
        else:
            print("  戦略該当馬なし")

    # ── 5. 人気帯別の精度(モデルが人気を超える妙味を出せているか) ──
    print("\n" + "=" * 55)
    print("5. 人気帯別 AI予測1位の的中(人気馬偏重でないか)")
    print("=" * 55)
    if "予測順位" in df.columns and "人気" in df.columns:
        top1 = df[df["予測順位"] == 1].copy()
        top1["人気帯"] = pd.cut(top1["人気"], [0, 1, 3, 6, 100],
                              labels=["1人気", "2-3人気", "4-6人気", "7人気以下"])
        for band, g in top1.groupby("人気帯", observed=True):
            if len(g) == 0:
                continue
            win = (g["着順_num"] == 1).mean() * 100
            print(f"  AI1位が{band}: {len(g)}回  勝率{win:.1f}%")

    # ── 6. 3モデルの的中検証(連対モデル・複勝モデルが機能しているか) ──
    print("\n" + "=" * 55)
    print("6. 3モデルの的中検証(連対・複勝モデルの実力)")
    print("=" * 55)
    # 連対順位1位 → 実際に連対(2着以内)したか
    if "連対順位" in df.columns:
        p2top = df[df["連対順位"] == 1]
        if len(p2top) > 0:
            ren = (p2top["着順_num"] <= 2).mean() * 100
            print(f"  連対モデル1位({len(p2top)}レース): 実連対率{ren:.1f}%")
    # 複勝順位1位 → 実際に複勝(3着以内)したか
    if "複勝順位" in df.columns:
        p3top = df[df["複勝順位"] == 1]
        if len(p3top) > 0:
            fuku = (p3top["着順_num"] <= 3).mean() * 100
            print(f"  複勝モデル1位({len(p3top)}レース): 実複勝率{fuku:.1f}%")
    # 勝ちモデル1位と複勝モデル1位が「別の馬」になっているか
    if "予測順位" in df.columns and "複勝順位" in df.columns:
        merged = df[(df["予測順位"] == 1) | (df["複勝順位"] == 1)]
        diff_races = 0
        total_races = 0
        for rid, g in merged.groupby("race_id"):
            win1 = g[g["予測順位"] == 1]
            fuku1 = g[g["複勝順位"] == 1]
            if len(win1) > 0 and len(fuku1) > 0:
                total_races += 1
                if win1["馬名"].iloc[0] != fuku1["馬名"].iloc[0]:
                    diff_races += 1
        if total_races > 0:
            print(f"  勝ち1位と複勝1位が別馬: {diff_races}/{total_races}レース "
                  f"({diff_races/total_races*100:.1f}%)")
            print("    → 高いほど複勝モデルが独自の視点を持っている証拠")

    # ── 7. 券種推奨の的中検証 ──
    print("\n" + "=" * 55)
    print("7. 券種推奨の的中検証(軸◎/相手○/穴▲)")
    print("=" * 55)
    if "券種推奨" in df.columns:
        rec = df[df["券種推奨"].notna() & (df["券種推奨"].astype(str) != "")
                 & (df["券種推奨"].astype(str) != "nan")]
        for role in ["軸◎", "軸(人気)", "相手○", "穴▲"]:
            sub = rec[rec["券種推奨"] == role]
            if len(sub) == 0:
                continue
            win = (sub["着順_num"] == 1).mean() * 100
            ren = (sub["着順_num"] <= 2).mean() * 100
            fuku = (sub["着順_num"] <= 3).mean() * 100
            print(f"  {role}: {len(sub):3d}頭  勝率{win:5.1f}%  "
                  f"連対率{ren:5.1f}%  複勝率{fuku:5.1f}%")
    else:
        print("  券種推奨列なし(新モデルの予想データが必要)")

    # ── 8. 推奨買い目の実払戻照合（today_bets.csv × payout） ──
    print("\n" + "=" * 55)
    print("8. 推奨買い目の実払戻照合（回収率検証）")
    print("=" * 55)
    bets_path = os.path.join(BASE_DIR, "today_bets.csv")
    if os.path.exists(bets_path):
        try:
            from payout_scraper import get_payout, _close_fallback_driver
            bets = pd.read_csv(bets_path, dtype={"race_id": str, "組み合わせ": str})
            pay_map = {}
            for rid in sorted(bets["race_id"].unique()):
                res = get_payout(rid)
                if res == "BLOCKED" or not res:
                    print(f"  払戻取得できず: {rid}")
                    time.sleep(1)
                    continue
                for r in res:
                    combo = str(r["組み合わせ"])
                    if r["券種"] in ("馬連", "ワイド", "3連複", "枠連"):
                        combo = "-".join(sorted(combo.split("-")))
                    pay_map[(rid, r["券種"], combo)] = r["払戻金"]
                time.sleep(0.4)
            _close_fallback_driver()

            def _bet_key(row):
                combo = str(row["組み合わせ"])
                if row["券種"] in ("馬連", "ワイド", "3連複"):
                    combo = "-".join(sorted(combo.split("-")))
                return (row["race_id"], row["券種"], combo)

            bets["払戻"] = bets.apply(lambda r: pay_map.get(_bet_key(r), 0), axis=1)
            bets["的中"] = bets["払戻"] > 0
            total_n, total_ret = len(bets), bets["払戻"].sum()
            print(f"  総点数{total_n}  的中{int(bets['的中'].sum())}  "
                  f"全体回収率{total_ret/(total_n*100)*100:.1f}%（100円均等買い想定）")
            print(f"  {'買い方':20} {'点数':>5} {'的中率':>7} {'回収率':>8}  {'BT':>6}")
            for name, g2 in bets.groupby("買い方"):
                roi = g2["払戻"].sum() / (len(g2) * 100) * 100
                bt = g2["BT回収率"].iloc[0] if "BT回収率" in g2.columns else "-"
                print(f"  {str(name):20} {len(g2):5d}  {g2['的中'].mean()*100:5.1f}%  "
                      f"{roi:6.1f}%  {bt:>4}%")

            # bets_result_log.csv に日次追記（同日上書き）
            blog = os.path.join(BASE_DIR, "bets_result_log.csv")
            today_s = datetime.now().strftime("%Y-%m-%d")
            log_rows = []
            for name, g2 in bets.groupby("買い方"):
                log_rows.append({"日付": today_s, "買い方": name, "点数": len(g2),
                                 "的中数": int(g2["的中"].sum()),
                                 "回収率": round(g2["払戻"].sum() / (len(g2) * 100) * 100, 1)})
            log_rows.append({"日付": today_s, "買い方": "＝合計＝", "点数": total_n,
                             "的中数": int(bets["的中"].sum()),
                             "回収率": round(total_ret / (total_n * 100) * 100, 1)})
            blog_df = pd.DataFrame(log_rows)
            if os.path.exists(blog):
                _ex = pd.read_csv(blog)
                _ex = _ex[_ex["日付"] != today_s]
                blog_df = pd.concat([_ex, blog_df], ignore_index=True)
            blog_df.to_csv(blog, index=False, encoding="utf-8-sig")
            print(f"  買い目ログ保存: {blog}")
        except Exception as e:
            print(f"  買い目照合エラー: {e}")
    else:
        print("  today_bets.csv なし（新ポリシーでの予想実行後に照合できます）")

    print("\n分析完了")

    # ── accuracy_log.csv に日次サマリーを追記 ──
    log_row = {"日付": datetime.now().strftime("%Y-%m-%d"), "レース数": len(df["race_id"].unique())}

    if "推奨ランク" in df.columns:
        honmei = df[df["推奨ランク"] == "◎"]
        log_row["◎勝率"]   = round((honmei["着順_num"] == 1).mean() * 100, 1) if len(honmei) else None
        log_row["◎複勝率"] = round((honmei["着順_num"] <= 3).mean() * 100, 1) if len(honmei) else None
        log_row["◎頭数"]   = len(honmei)

    if "予測順位" in df.columns:
        top1 = df[df.groupby("race_id")["予測順位"].transform("min") == df["予測順位"]]
        log_row["AI1位勝率"]   = round((top1["着順_num"] == 1).mean() * 100, 1) if len(top1) else None
        log_row["AI1位複勝率"] = round((top1["着順_num"] <= 3).mean() * 100, 1) if len(top1) else None

    if "MF勝ち確率" in df.columns:
        mf_idx = df.groupby("race_id")["MF勝ち確率"].idxmax().dropna()
        mf_top = df.loc[mf_idx]
        log_row["MF1位勝率"]   = round((mf_top["着順_num"] == 1).mean() * 100, 1) if len(mf_top) else None
        log_row["MF1位複勝率"] = round((mf_top["着順_num"] <= 3).mean() * 100, 1) if len(mf_top) else None

    if "単勝期待値" in df.columns:
        ev_hits = df[df["単勝期待値"] >= 0.4]
        log_row["EV0.4以上頭数"] = len(ev_hits)
        log_row["EV0.4以上勝率"] = round((ev_hits["着順_num"] == 1).mean() * 100, 1) if len(ev_hits) else None

    log_df = pd.DataFrame([log_row])
    if os.path.exists(LOG_FILE):
        existing = pd.read_csv(LOG_FILE)
        existing = existing[existing["日付"] != log_row["日付"]]  # 同日は上書き
        log_df = pd.concat([existing, log_df], ignore_index=True)
    log_df.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")
    print(f"  ログ保存: {LOG_FILE}")


if __name__ == "__main__":
    main()