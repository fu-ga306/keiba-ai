# -*- coding: utf-8 -*-
"""バックテストの罠を機械的に検査する（2026-08-27）

なぜ作ったか
  自分の競馬AIで「回収率100%超」を8回作って8回とも失った。原因は6種類で、
  どれも**エラーを出さずに静かに数字を作る**タイプだった。
  同じ検査を毎回手でやるのは無理があるので、機械化する。

  「判断する」のではなく「実行して結果を見る」形にすることで、
  ・毎回の目視が要らなくなる
  ・根拠が再現可能になる（クライアントにも同じスクリプトを渡せる）

⚠ このツールの限界（先に読んでください）
  検査できるのは **既知の6種類の罠だけ** です。7種類目は見つけられません。
  また静的検査は「疑わしい箇所」を挙げるだけで、**クロなのは人が判断します**。
  「このツールが通ったから安全」ではありません。「既知の罠は踏んでいない」だけです。

使い方
  python leak_audit.py --code <ディレクトリ>       コードの静的検査
  python leak_audit.py --data <csv> [--id 列] [--date 列]  データの検査
  python leak_audit.py --sample <csv> --payoff 列  何点あれば何が言えるか
  python leak_audit.py --all <ディレクトリ>        コード検査をまとめて
"""
import sys

for _s in (sys.stdout, sys.stderr):   # cp932環境でのUnicodeEncodeError→異常終了を防ぐ
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import os
import re

FINDINGS = []


def log(m):
    print(m, flush=True)


def note(level, trap, msg, where=""):
    """所見を1件記録する。level: 高/中/低"""
    FINDINGS.append({"level": level, "trap": trap, "msg": msg, "where": where})


# ── コードの静的検査 ────────────────────────────────────────────────
#   ⚠ ここで挙がるのは「疑わしい箇所」であって、クロと決まったわけではない。
#     ただし、私が実際に踏んだ罠はすべてこのパターンに合致した。
#   ⚠ 2026-08-27: 最初の版は自分のプロジェクトに当てたら1,326件出た。
#     うち822件は「列名に『確定』が入っている」だけの検出で、コメントや
#     docstringまで拾っていた。**1,326件の所見は読む人にとってノイズであり、
#     報告書として無価値**（オオカミ少年になり、本当の問題が埋もれる）。
#     そこで方針を変えた:
#       ・確実に危ないパターンだけを【高】として個別に挙げる
#       ・広く当たるパターンは件数だけ集計し、代表例を数件出す
#       ・1パターンあたりの個別列挙は上限を設ける
PATTERNS = [
    # ── 【高】確実に危ない。個別に挙げる ──
    (r"sort_values\s*\(\s*\[?\s*[\"'][^\"']*(?:_id|id_|ID|Id)\b[^\"']*[\"']",
     "高", "罠1", "IDでソートしています。IDの大小が時系列と一致するか確認してください", True),
    (r"strptime\s*\([^)]*\[\s*:\s*\d+\s*\]",
     "高", "罠1", "IDの先頭を切って日付にしています。桁が合えば例外が出ないので気づけません", True),
    (r"train_test_split\s*\((?![^)]*shuffle\s*=\s*False)",
     "高", "罠1", "train_test_split をそのまま使っています。時系列ならシャッフル不可です", True),

    # ── 【中】文脈しだい。件数を集計し、代表例だけ出す ──
    (r"\.sample\s*\(\s*frac\s*=",
     "中", "罠1", "ランダム抽出で分割していませんか。時系列なら時点で切る必要があります", False),
    (r"itertools\.product",
     "中", "罠3", "グリッド探索があります。順列検定で偽物の水準と比べてください", False),
]

# read_csv の dtype 未指定は、**同じファイルが結合もしている場合だけ**意味がある。
# 単独で挙げると数百件になり、ノイズにしかならない。
_RE_READ_NO_DTYPE = re.compile(r"read_csv\s*\((?![^)]*dtype)")
_RE_JOIN = re.compile(r"\.merge\s*\(|\.join\s*\(|\.map\s*\(|pd\.concat\s*\(")
_MAX_LIST = 8          # 1パターンあたり、個別に列挙する上限

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".mypy_cache"}


def audit_code(root):
    log("=" * 66)
    log("  コードの静的検査")
    log("=" * 66)
    log("  ⚠ 挙がるのは『疑わしい箇所』です。クロかどうかは人が判断します。")
    files = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        files += [os.path.join(dp, f) for f in fns if f.endswith((".py", ".ipynb"))]
    if not files:
        log(f"  対象ファイルがありません: {root}")
        return
    log(f"  対象 {len(files)}ファイル\n")

    per_pat = {}          # パターンごとに (件数, 例のリスト) を貯める
    join_risk = []        # 結合していて、かつ dtype 未指定で読んでいるファイル
    for fp in files:
        try:
            src = open(fp, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        rel = os.path.relpath(fp, root)
        lines = src.splitlines()
        for pat, lv, trap, msg, each in PATTERNS:
            for m in re.finditer(pat, src):
                ln = src[:m.start()].count("\n") + 1
                code = lines[ln - 1].strip()[:70] if ln <= len(lines) else ""
                k = (lv, trap, msg, each)
                c, ex = per_pat.get(k, (0, []))
                if len(ex) < _MAX_LIST:
                    ex.append((f"{rel}:{ln}", code))
                per_pat[k] = (c + 1, ex)
        # dtype 未指定 × 結合あり の組み合わせだけを見る
        if _RE_READ_NO_DTYPE.search(src) and _RE_JOIN.search(src):
            join_risk.append(rel)

    if not per_pat and not join_risk:
        log("  既知のパターンに一致する箇所はありませんでした。")
        log("")
        return

    for (lv, trap, msg, each), (cnt, ex) in sorted(
            per_pat.items(), key=lambda x: {"高": 0, "中": 1, "低": 2}[x[0][0]]):
        note(lv, trap, f"{msg}（{cnt}箇所）", ex[0][0] if ex else "")
        log(f"  [{lv}] {trap} {cnt}箇所 — {msg}")
        show = ex if each else ex[:3]
        for where, code in show:
            log(f"        {where}")
            log(f"          {code}")
        if cnt > len(show):
            log(f"        …他 {cnt - len(show)}箇所")
        log("")

    if join_risk:
        note("中", "罠5",
             f"dtype未指定で読み、かつ結合しているファイルが{len(join_risk)}件",
             join_risk[0])
        log(f"  [中] 罠5 {len(join_risk)}ファイル — dtype 未指定で読み、かつ結合しています")
        log("        結合キーの0埋めが消えると、一致せず静かに件数が減ります")
        for f in join_risk[:5]:
            log(f"        {f}")
        if len(join_risk) > 5:
            log(f"        …他 {len(join_risk)-5}件")
        log("")


# ── データの検査 ───────────────────────────────────────────────────
def audit_data(path, id_col=None, date_col=None):
    import pandas as pd
    log("=" * 66)
    log("  データの検査")
    log("=" * 66)
    try:
        df = pd.read_csv(path, dtype=str, nrows=200000)
    except Exception as e:
        log(f"  読めません: {type(e).__name__}: {e}")
        return
    log(f"  {os.path.basename(path)}  {len(df):,}行 / {len(df.columns)}列\n")

    # ① 0埋めが消えていないか（型を指定せずに読むと消える列を探す）
    log("  ① 結合キーの0埋め（罠5）")
    loose = pd.read_csv(path, nrows=5000)
    risky = []
    for c in df.columns:
        v = df[c].dropna().astype(str)
        if v.empty:
            continue
        has_zero = (v.str.match(r"^0\d+$")).any()
        became_num = c in loose.columns and str(loose[c].dtype).startswith(("int", "float"))
        if has_zero and became_num:
            risky.append(c)
    if risky:
        for c in risky:
            note("高", "罠5", "0埋めが数値化で消えます。dtype=str で読んでください", f"{path}:{c}")
            log(f"     [高] 列 '{c}' … 0埋めが消えます（例: '03' → 3）")
    else:
        log("     問題なし")

    # ② IDでソートしたとき時系列になるか
    log("\n  ② IDの順序＝時系列か（罠1）")
    if id_col and date_col and id_col in df.columns and date_col in df.columns:
        d = df[[id_col, date_col]].dropna().copy()
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d.dropna().sort_values(id_col)
        ok = d[date_col].is_monotonic_increasing
        if ok:
            log(f"     ○ '{id_col}' でソートすると '{date_col}' が単調増加します")
        else:
            bad = int((d[date_col].diff().dt.total_seconds() < 0).sum())
            note("高", "罠1", f"IDでソートしても時系列にならない（逆行{bad}件）",
                 f"{path}:{id_col}")
            log(f"     [高] '{id_col}' でソートしても時系列になりません（逆行 {bad}件）")
            log(f"          IDを時系列として使っているなら、学習側に未来が混ざります")
    else:
        log("     --id と --date を指定すると検査できます")

    # ③ IDの先頭を日付として読めてしまうか
    log("\n  ③ IDが日付に化けないか（罠1）")
    checked = False
    for c in df.columns:
        v = df[c].dropna().astype(str)
        if v.empty or not v.str.match(r"^\d{8,}$").all():
            continue
        checked = True
        head = pd.to_datetime(v.str[:8], format="%Y%m%d", errors="coerce")
        rate = head.notna().mean()
        if rate > 0.9:
            msg = f"列 '{c}' は先頭8桁が日付として解釈できます（{rate*100:.0f}%）"
            if date_col and date_col in df.columns:
                real = pd.to_datetime(df[date_col], errors="coerce")
                agree = (head.dt.date == real.dt.date).mean()
                if agree < 0.9:
                    note("高", "罠1", f"{msg}。しかし実際の日付とは{agree*100:.0f}%しか一致しません",
                         f"{path}:{c}")
                    log(f"     [高] {msg}")
                    log(f"          実際の日付との一致は {agree*100:.0f}%。**日付ではありません**")
                    continue
            note("中", "罠1", msg + "。例外が出ないので誤用に気づけません", f"{path}:{c}")
            log(f"     [中] {msg}")
            log(f"          パースが必ず成功するため、誤用しても例外が出ません")
    if not checked:
        log("     数値のみのID列がありません")
    log("")


# ── 標本設計 ───────────────────────────────────────────────────────
def audit_sample(path, payoff_col):
    import numpy as np
    import pandas as pd
    log("=" * 66)
    log("  標本設計（罠6）― 何点あれば何が言えるか")
    log("=" * 66)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        log(f"  読めません: {type(e).__name__}: {e}")
        return
    if payoff_col not in df.columns:
        log(f"  列 '{payoff_col}' がありません。候補: {list(df.columns)[:10]}")
        return
    v = pd.to_numeric(df[payoff_col], errors="coerce").dropna().values
    if len(v) < 30:
        log(f"  {len(v)}件では計算できません")
        return
    rng = np.random.default_rng(20260827)
    log(f"  検体 {len(v):,}件  平均 {v.mean():.1f}  標準偏差 {v.std():.1f}")
    log(f"  中央値 {np.median(v):.1f}  ゼロの割合 {(v == 0).mean()*100:.1f}%\n")
    if v.std() > v.mean() * 2:
        note("高", "罠6", "分散が平均に対して大きく、少数の当たりが平均を支えています", path)
        log("  [高] 分散が大きい構造です。少数の当たりが平均を作っています")
        log("       → 短期の平均値にはほとんど情報がありません\n")
    log(f"  {'点数':>8}{'95%下限':>10}{'95%上限':>10}")
    prev = None
    for n in (100, 200, 400, 800, 1600, 3200, 6400, 12800):
        s = np.array([rng.choice(v, n).mean() for _ in range(2000)])
        lo, hi = np.percentile(s, [2.5, 97.5])
        mark = ""
        if prev is None or (prev <= 100 < lo):
            mark = "  ← ここで100を上回る" if lo > 100 else ""
        log(f"  {n:>8,}{lo:>10.1f}{hi:>10.1f}{mark}")
        prev = lo
        if lo > 100:
            break
    else:
        log("\n  12,800点でも95%下限が100を超えません。")
        log("  **この指標で優位を証明するのは現実的でない**という意味です。")
        note("高", "罠6", "現実的な標本数では優位を証明できません", path)
    log("")


def summary():
    log("=" * 66)
    log("  まとめ")
    log("=" * 66)
    if not FINDINGS:
        log("  既知の罠に該当する所見はありませんでした。")
    else:
        for lv in ("高", "中", "低"):
            xs = [f for f in FINDINGS if f["level"] == lv]
            if not xs:
                continue
            log(f"\n  【{lv}】{len(xs)}件")
            seen = set()
            for f in xs:
                k = (f["trap"], f["msg"])
                if k in seen:
                    continue
                seen.add(k)
                n = sum(1 for g in xs if (g["trap"], g["msg"]) == k)
                log(f"    - {f['trap']} {f['msg']}" + (f"（{n}箇所）" if n > 1 else ""))
                log(f"      例: {f['where']}")
    log("\n  ⚠ このツールが検査したのは既知の6種類だけです。")
    log("    『通ったから安全』ではなく『既知の罠は踏んでいない』という意味です。")
    log("    特に次の2つは機械では確かめられません。人がやってください。")
    log("      ・順列検定（探索ごと再現する必要がある）")
    log("      ・検証と本番で同じ関数を使っているか")


def main():
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--code", help="コードを静的検査するディレクトリ")
    p.add_argument("--data", help="検査するCSV")
    p.add_argument("--id", help="ID列の名前（--data と併用）")
    p.add_argument("--date", help="日付列の名前（--data と併用）")
    p.add_argument("--sample", help="標本設計に使うCSV")
    p.add_argument("--payoff", help="払戻・損益の列名（--sample と併用）")
    p.add_argument("--all", help="ディレクトリを指定してコード検査")
    a = p.parse_args()

    if not any([a.code, a.data, a.sample, a.all]):
        log(__doc__)
        return
    if a.code or a.all:
        audit_code(a.code or a.all)
    if a.data:
        audit_data(a.data, a.id, a.date)
    if a.sample:
        if not a.payoff:
            log("  --sample には --payoff で列名を指定してください")
        else:
            audit_sample(a.sample, a.payoff)
    summary()


if __name__ == "__main__":
    main()
