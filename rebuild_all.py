# -*- coding: utf-8 -*-
"""直した元データから、特徴量・モデル・検証データを作り直す（2026-09-02）

なぜ要るか
  race_data_clean.csv のクラス・回り・賞金を直したが、
  そこから作る race_features.csv / model_resid.pkl / resid_kinds_pred.csv は
  古い（壊れたデータで作った）ままになっている。
  作り直さないと、直した意味がモデルに届かない。

  週次更新でも同じことをするが、次は9/8。待たずにここで回す。

⚠ 重い。予想システムが動いていない時間に走らせること。
  終わったらメールで知らせる。
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

STEPS = [
    ("特徴量の再生成", ["-c",
     "import sys; sys.path.insert(0, r'" + BASE_DIR + "'); "
     "from features import build_features; build_features()"], 7200),
    ("残差モデルの再学習", ["train_resid.py"], 7200),
    ("検証データの作り直し", ["resid_kinds.py"], 14400),
    ("実装の照合", ["check_resid.py"], 3600),
]


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def main():
    # 予想が動いていたら走らせない。メモリを奪い合って両方失敗する
    try:
        import psutil
        for p in psutil.process_iter(["cmdline"]):
            cl = p.info["cmdline"] or []
            # ⚠ 文字列の部分一致で見ないこと（2026-09-02に踏んだ）
            #   確認用に打ったコマンドの中身に 'auto_predict_publish' が
            #   含まれていて、自分自身に反応して中止した。
            #   **引数として実際にそのスクリプトを走らせているか**を見る。
            if any(str(a).replace("\\", "/").endswith("auto_predict_publish.py")
                   for a in cl):
                # 開催日でなければ予想は待機しているだけなので走らせてよい。
                # 開催日（today_predictions.csv が当日更新）のときだけ止める。
                import os as _os, datetime as _dt
                _tp = _os.path.join(BASE_DIR, "today_predictions.csv")
                _race_day = (_os.path.exists(_tp) and
                             _dt.date.fromtimestamp(_os.path.getmtime(_tp))
                             == _dt.date.today())
                if _race_day:
                    log(f"⚠ 開催日で予想が稼働中（pid{p.pid}）。中止します")
                    return
                log(f"  予想は稼働中（pid{p.pid}）だが開催日ではないので続行します")
                break
    except Exception:
        pass

    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    out, t0 = [], datetime.now()
    for name, args, tmo in STEPS:
        s0 = datetime.now()
        cmd = [PYTHON] + ([args[0]] if args[0] == "-c" else
                          [os.path.join(BASE_DIR, args[0])])
        if args[0] == "-c":
            cmd = [PYTHON, "-c", args[1]]
        log(f"{name} 開始")
        try:
            r = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               timeout=tmo, env=env)
            ok = r.returncode == 0
            el = (datetime.now() - s0).total_seconds() / 60
            log(f"{name} {'成功' if ok else '失敗'}  {el:.0f}分")
            out.append((name, ok, el, (r.stdout or "")[-2500:], (r.stderr or "")[-1200:]))
            if not ok:
                log("  以降を中止します（前段が失敗すると後段は意味を持たない）")
                break
        except subprocess.TimeoutExpired:
            log(f"{name} 打ち切り")
            out.append((name, False, tmo / 60, "", "タイムアウト"))
            break

    body = [f"直したデータから作り直しました。所要 "
            f"{(datetime.now()-t0).total_seconds()/60:.0f}分", ""]
    body.append("■ 見るところ")
    body.append("  check_resid の ROI が 119.2% からどう動いたか。")
    body.append("  買い方は一切変えていないので、動いた分はすべて")
    body.append("  クラス・回り・賞金が埋まった効果です。")
    body.append("")
    for name, ok, el, so, se in out:
        body.append("=" * 60)
        body.append(f"■ {name}  {'成功' if ok else '失敗'}  {el:.0f}分")
        body.append("=" * 60)
        if se.strip():
            body.append("--- stderr ---")
            body.append(se)
        body.append(so)
        body.append("")
    try:
        import auto_predict_publish as A
        mark = "○" if all(o[1] for o in out) else "⚠"
        A._send_alert(f"【競馬AI】{mark} 特徴量・モデルの作り直し", "\n".join(body))
        log("メールを送信しました")
    except Exception as e:
        log(f"メール送信に失敗: {e}")


if __name__ == "__main__":
    main()
