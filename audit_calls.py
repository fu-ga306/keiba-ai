# -*- coding: utf-8 -*-
"""「呼び出しが足りない」型の穴を探す（2026-08-18）

なぜ作るか
  update_data.py が build_sire_stats() を引数なしで1回だけ呼び、本番用しか
  作られていなかった。学習用は1か月半止まっていた。
  スクリプトを直接叩けば両方できるのに、関数を import して呼ぶ経路だけが
  片方しかやっていなかった。

  同じ形の穴を探す。「__main__ でやっていることを、import 経由の呼び出しが
  再現できていない」箇所が危ない。

見るもの
  ① __main__ で複数回呼ばれている関数が、他所では1回しか呼ばれていないか
  ② 出力ファイルを2つ以上作るスクリプトで、片方しか参照されていないか
  ③ 引数の既定値に頼った呼び出し（suffix や max_year を省略しているもの）

実行: python audit_calls.py
"""
import ast
import os
import re
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
# 監査対象（本番の運用に関わるものだけ。検証用スクリプトは除く）
CORE = ["update_data.py", "weekly_update.py", "auto_predict_publish.py",
        "keiba_predict.py", "features.py", "sire_stats.py", "train_mf_v2.py",
        "train_resid.py", "build_calibrator.py", "build_grade.py",
        "prep_cache.py", "result_tracker.py", "flask_app.py"]


def log(m):
    print(m, flush=True)


def calls_in(tree, name):
    """関数名 name の呼び出しを (行, 引数の数, キーワード名) で返す。"""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            nm = f.id if isinstance(f, ast.Name) else (
                f.attr if isinstance(f, ast.Attribute) else None)
            if nm == name:
                out.append((n.lineno, len(n.args),
                            sorted(k.arg for k in n.keywords if k.arg)))
    return out


def main():
    log("=" * 70)
    log("  ① __main__ で複数回呼ぶ関数を、他所が1回しか呼んでいないか")
    log("=" * 70)
    # 各ファイルの __main__ ブロックで複数回呼ばれる関数を集める
    multi = {}
    for f in CORE:
        p = os.path.join(BASE, f)
        if not os.path.exists(p):
            continue
        try:
            tree = ast.parse(open(p, encoding="utf-8").read())
        except Exception:
            continue
        for n in tree.body:
            if not (isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                    and getattr(n.test.left, "id", "") == "__name__"):
                continue
            cnt = defaultdict(list)
            for c in ast.walk(n):
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name):
                    cnt[c.func.id].append(
                        (c.lineno, sorted(k.arg for k in c.keywords if k.arg)))
            for fn, xs in cnt.items():
                if len(xs) >= 2:
                    multi[(f, fn)] = xs

    if not multi:
        log("  該当なし")
    for (f, fn), xs in multi.items():
        log(f"\n  {f} の __main__ は {fn}() を {len(xs)}回呼ぶ")
        for ln, kw in xs:
            log(f"    行{ln}: 引数 {kw if kw else '(なし)'}")
        # 他のファイルでの呼び出しを調べる
        for g in CORE:
            if g == f:
                continue
            gp = os.path.join(BASE, g)
            if not os.path.exists(gp):
                continue
            try:
                gt = ast.parse(open(gp, encoding="utf-8").read())
            except Exception:
                continue
            cs = calls_in(gt, fn)
            if not cs:
                continue
            if len(cs) < len(xs):
                log(f"    ⚠ {g} は {len(cs)}回しか呼んでいない"
                    f"（{len(xs)}回必要）")
                for ln, na, kw in cs:
                    log(f"       行{ln}: 引数 {kw if kw else '(なし)'}")
            else:
                log(f"    ○ {g} も {len(cs)}回呼んでいる")

    log("\n" + "=" * 70)
    log("  ② 出力を複数作るのに、片方しか使われていないファイル")
    log("=" * 70)
    # コード全体で参照されているファイル名を集める
    ref = defaultdict(set)
    for f in os.listdir(BASE):
        if not f.endswith(".py"):
            continue
        try:
            t = open(os.path.join(BASE, f), encoding="utf-8").read()
        except Exception:
            continue
        for m in re.findall(r'"([A-Za-z0-9_]+\.(?:csv|pkl))"', t):
            ref[m].add(f)
    PAIRS = [("sire_stats_father.csv", "sire_stats_father_train.csv"),
             ("sire_stats_bms.csv", "sire_stats_bms_train.csv"),
             ("model_mf.pkl", "model_mf_bt.pkl")]
    for a, b in PAIRS:
        ra, rb = len(ref.get(a, ())), len(ref.get(b, ()))
        log(f"  {a:<32}{ra}ファイルが参照")
        log(f"  {b:<32}{rb}ファイルが参照"
            + ("  ⚠ 参照が少ない" if rb and rb < ra / 2 else ""))

    log("\n" + "=" * 70)
    log("  ③ 引数を省略した呼び出し（既定値に頼っている）")
    log("=" * 70)
    log("  既定値のまま呼ぶと、片方のスナップショットしか作られないなどの穴になる。")
    WATCH = {"build_sire_stats": ["max_year", "suffix"],
             "build_features": ["year_max"],
             "build_sire_stats_from": ["max_year"]}
    for f in CORE:
        p = os.path.join(BASE, f)
        if not os.path.exists(p):
            continue
        try:
            tree = ast.parse(open(p, encoding="utf-8").read())
        except Exception:
            continue
        for fn, need in WATCH.items():
            for ln, na, kw in calls_in(tree, fn):
                miss = [k for k in need if k not in kw]
                if miss and na == 0:
                    log(f"  ⚠ {f}:{ln}  {fn}() が {miss} を省略")
                elif not miss:
                    log(f"  ○ {f}:{ln}  {fn}({', '.join(kw)})")

    log("\n" + "=" * 70)
    log("  まとめ")
    log("=" * 70)
    log("  ⚠ が付いたものは、片方しか作られていない／再現できていない可能性がある。")
    log("  実際に 2026-08-18 に update_data.py の build_sire_stats() で発生した。")


if __name__ == "__main__":
    main()
