with open(r'c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai\model.py', 'r', encoding='utf-8') as f:
    content = f.read()

# クラス別バックテストの後に戦略F・G・FGを追加
old = '''    for label, n, rate, roi in sorted(cls_results, key=lambda x: x[3], reverse=True):
            mark = "🟢" if roi >= 120 else "🟡" if roi >= 100 else "🔴"
            print(f"  {mark} {label:8}: {n:3}回 {rate:.1f}% 回収率{roi:.1f}%")'''

new = '''    for label, n, rate, roi in sorted(cls_results, key=lambda x: x[3], reverse=True):
            mark = "🟢" if roi >= 120 else "🟡" if roi >= 100 else "🔴"
            print(f"  {mark} {label:8}: {n:3}回 {rate:.1f}% 回収率{roi:.1f}%")

    # ── ⑥ 新戦略F・G・FG ────────────────────────────────────────────────
    print(f"\\n{'='*40}\\n🆕 新戦略バックテスト\\n{'='*40}")

    # 戦略F：中京・東京限定 × 戦略A
    # 競馬場cd: 5=東京, 7=中京
    if "競馬場cd" in df.columns:
        bets_f = df[
            (df["競馬場cd"].isin([5, 7]))
            & (df["予測順位"] == 1)
            & (df["単勝期待値"] >= 0.3)
            & (df["単勝オッズ"] >= 1.5)
            & (df["単勝オッズ"] <= 20.0)
        ]
        if len(bets_f) > 0:
            wins_f = bets_f[bets_f["着順_num"] == 1]
            roi_f  = wins_f["単勝オッズ"].sum() / len(bets_f) * 100
            print(f"\\n── 戦略F: 中京・東京 × 予測1位 × 期待値>=0.3 ──")
            print(f"ベット数: {len(bets_f)}回")
            print(f"的中数:   {len(wins_f)}回")
            print(f"的中率:   {len(wins_f)/len(bets_f)*100:.1f}%")
            print(f"回収率:   {roi_f:.1f}%")

    # 戦略G：短距離限定（〜1400m）× 戦略A
    if "距離" in df.columns:
        bets_g = df[
            (df["距離"] <= 1400)
            & (df["予測順位"] == 1)
            & (df["単勝期待値"] >= 0.3)
            & (df["単勝オッズ"] >= 1.5)
            & (df["単勝オッズ"] <= 20.0)
        ]
        if len(bets_g) > 0:
            wins_g = bets_g[bets_g["着順_num"] == 1]
            roi_g  = wins_g["単勝オッズ"].sum() / len(bets_g) * 100
            print(f"\\n── 戦略G: 短距離(〜1400m) × 予測1位 × 期待値>=0.3 ──")
            print(f"ベット数: {len(bets_g)}回")
            print(f"的中数:   {len(wins_g)}回")
            print(f"的中率:   {len(wins_g)/len(bets_g)*100:.1f}%")
            print(f"回収率:   {roi_g:.1f}%")

    # 戦略FG：中京・東京 × 短距離（最強組み合わせ）
    if "競馬場cd" in df.columns and "距離" in df.columns:
        bets_fg = df[
            (df["競馬場cd"].isin([5, 7]))
            & (df["距離"] <= 1400)
            & (df["予測順位"] == 1)
            & (df["単勝期待値"] >= 0.3)
            & (df["単勝オッズ"] >= 1.5)
            & (df["単勝オッズ"] <= 20.0)
        ]
        if len(bets_fg) > 0:
            wins_fg = bets_fg[bets_fg["着順_num"] == 1]
            roi_fg  = wins_fg["単勝オッズ"].sum() / len(bets_fg) * 100
            print(f"\\n── 戦略FG: 中京・東京 × 短距離 × 予測1位 × 期待値>=0.3 ──")
            print(f"ベット数: {len(bets_fg)}回")
            print(f"的中数:   {len(wins_fg)}回")
            print(f"的中率:   {len(wins_fg)/len(bets_fg)*100:.1f}%")
            print(f"回収率:   {roi_fg:.1f}%")

    # 戦略H：小倉・中山 × 戦略A
    if "競馬場cd" in df.columns:
        bets_h = df[
            (df["競馬場cd"].isin([6, 10]))
            & (df["予測順位"] == 1)
            & (df["単勝期待値"] >= 0.3)
            & (df["単勝オッズ"] >= 1.5)
            & (df["単勝オッズ"] <= 20.0)
        ]
        if len(bets_h) > 0:
            wins_h = bets_h[bets_h["着順_num"] == 1]
            roi_h  = wins_h["単勝オッズ"].sum() / len(bets_h) * 100
            print(f"\\n── 戦略H: 中山・小倉 × 予測1位 × 期待値>=0.3 ──")
            print(f"ベット数: {len(bets_h)}回")
            print(f"的中数:   {len(wins_h)}回")
            print(f"的中率:   {len(wins_h)/len(bets_h)*100:.1f}%")
            print(f"回収率:   {roi_h:.1f}%")'''

if old in content:
    content = content.replace(old, new, 1)
    with open(r'c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai\model.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("model.py 修正完了")
else:
    print("対象なし")
