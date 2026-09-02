"""
weekly_update.py
────────────────
毎週月曜日 08:00 に自動実行するスクリプト。

実行順序:
  0. レース結果スクレイピング（直近2週分）→ race_data_clean.csv 更新
  1. クリーニング（cleaner.py）
  2. sire_stats 再集計
  3. 特徴量再生成（features.py）
  4. モデル再学習（model.py + market_free_model.py）
  5. result_tracker.py update（予想記録の照合）
  6. GitHubにプッシュ（予想記録・モデル結果）

タスクスケジューラ登録済み：毎週月曜 08:00
"""

import os
import sys
import subprocess
from datetime import datetime, timedelta

BASE_DIR = r"c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai"
PYTHON   = r"C:/Users/別府飛河/AppData/Local/Microsoft/WindowsApps/python3.11.exe"
LOG_FILE = os.path.join(BASE_DIR, "weekly_update_log.txt")


def log(msg):
    ts = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


STEP_RESULTS = []   # (ラベル, 成否, 所要秒) を貯めて最後にメールで知らせる


def run_step(label, cmd, timeout=3600):
    """サブプロセスを実行してログに記録。失敗してもクラッシュしない。"""
    log(f"--- {label} 開始 ---")
    _t0 = datetime.now()
    try:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"  # Windows CP932 → UTF-8 強制（エンコードエラー防止）
        result = subprocess.run(
            cmd, cwd=BASE_DIR, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout,
            env=env
        )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if stdout:
            for line in stdout.splitlines()[-10:]:
                log(f"  {line}")
        if result.returncode != 0:
            log(f"  [警告] 終了コード {result.returncode}")
            if stderr:
                for line in stderr.splitlines()[-5:]:
                    log(f"  ERR: {line}")
        else:
            log(f"--- {label} 完了 ---")
        STEP_RESULTS.append((label, result.returncode == 0,
                             int((datetime.now() - _t0).total_seconds())))
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log(f"  [タイムアウト] {label} が {timeout}秒 を超過しました")
        STEP_RESULTS.append((label + "（時間切れ）", False,
                             int((datetime.now() - _t0).total_seconds())))
        return False
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log(f"  [エラー] {label}: {e}")
        for line in tb.splitlines()[-10:]:
            log(f"  TB: {line}")
        STEP_RESULTS.append((label + "（例外）", False,
                             int((datetime.now() - _t0).total_seconds())))
        return False


def _is_race_day():
    """今日が開催日か。ファイルの鮮度だけで判断し、スクレイピングはしない。

    週次は火曜に移したが、山の日などの祝日火曜には開催があり得る。
    開催日に走らせると Step0 のスクレイピングが当日の予想・結果取得と
    ぶつかり、ブロックを招く（2026-07-27に実際に400を食らっている）。
    """
    today = datetime.now().date()
    for fn in ("today_race_times.json", "today_predictions.csv"):
        p = os.path.join(BASE_DIR, fn)
        if os.path.exists(p) and \
                datetime.fromtimestamp(os.path.getmtime(p)).date() == today:
            return True
    return False


def main():
    log("=" * 50)
    log("週次自動更新 開始")
    log("=" * 50)

    # 開催日には走らせない（2026-08-09）。Step0は直近2週を取り直すので、
    # 1週飛ばしてもデータは失われない。翌週の実行で追いつく。
    if _is_race_day():
        log("本日は開催日 → 週次更新を見送ります（来週に実施）")
        STEP_RESULTS.append(("開催日のため見送り", True, 0))
        _notify(timedelta(0))
        return

    start = datetime.now()

    # ── Step 0: スクレイピング（直近2週分の新規レース取得） ───────────────
    log("[Step0] レース結果スクレイピング（直近2週）")
    run_step(
        "スクレイピング",
        [PYTHON, os.path.join(BASE_DIR, "update_data.py"),
         "--weeks", "2",
         "--skip-horse",      # horse_scraperは時間がかかるため別タスクで実施
         "--skip-features",   # features/modelは後ほど個別実行
         "--skip-model"],
        timeout=1800  # 30分
    )

    # ── Step 0.5: 血統マスタと種牡馬成績の更新（2026-08-18追加）────────────
    #   Step0 は --skip-horse で血統取得を飛ばしていたため、horse_master.csv が
    #   2026-07-08 で止まり、2026年の新馬598頭の父・母父が不明になっていた。
    #   血統はモデル寄与の7.4%を占めるので、欠けたままにしない。
    #
    #   ⚠ 順序が重要。horse_master → sire_stats → features の順でないと、
    #     新しい馬の血統が特徴量に反映されない。
    #   ⚠ horse_scraper は1頭あたり2.5〜4.5秒待つ。598頭で約35分。
    #     取得済みはスキップするので、次回以降は数分で終わる。
    #   ⚠ 失敗しても後続は続ける。血統が古いままでも予想は出せる。
    log("[Step0.5] 血統マスタ更新（未取得の馬だけ取得）")
    if not run_step("horse_scraper.py",
                    [PYTHON, os.path.join(BASE_DIR, "horse_scraper.py")],
                    timeout=7200):   # 2時間（新馬が大量に出る時期を考慮）
        log("  [警告] 血統の取得に失敗。前回の horse_master.csv のまま続けます")

    # ── Step 0.55: 払戻の取得（2026-08-22追加）──────────────────────────
    #   JRA-VANの契約を終了する方針なので、払戻の取得元をnetkeibaへ移す。
    #   過去分(2019-2026・26,051レース)は jv_payouts.csv にローカル保存済みで
    #   今後もそのまま使うため、取り直しは不要。新しいレースだけ取る。
    #
    #   ⚠ 払戻が無いと前向き検証の回収率が出せない。ここが止まると
    #     「記録は貯まるのに成績が分からない」状態になる。
    #   ⚠ 取得済み判定は payout_data.csv と jv_payouts.csv の両方を見る。
    #     片方しか見ないと2万件超を無駄に取りに行く（2026-08-22に修正）。
    log("[Step0.55] 払戻の取得（netkeiba・未取得分のみ）")
    if not run_step("payout_scraper.py",
                    [PYTHON, os.path.join(BASE_DIR, "payout_scraper.py")],
                    timeout=7200):
        log("  [警告] 払戻の取得に失敗。回収率の集計が古いままになります")

    log("[Step0.6] 種牡馬成績の再集計")
    if not run_step("sire_stats.py",
                    [PYTHON, os.path.join(BASE_DIR, "sire_stats.py")],
                    timeout=1800):
        log("  [警告] 種牡馬成績の再集計に失敗。前回の集計のまま続けます")

    # ── Step 0.7: 名寄せ表とコース表を作り直す ────────────────────────────
    #   どちらも race_data_clean.csv から作った表なので、
    #   元が伸びたら作り直さないと古くなる。
    #
    #   name_master.csv  出馬表の省略名 → 履歴の正式名
    #     ここが古いと、新しく乗り始めた騎手・開業した調教師の成績が
    #     欠損のままモデルに渡る。2026-08-30に、この不一致で
    #     騎手90.9%・調教師100%が欠損していたのを直した経緯がある。
    #
    #   course_turn.csv  競馬場・馬場・距離 → 回り
    #     出馬表から回りが取れなかったときの受け皿。
    log("[Step0.7] 名寄せ表・コース表の再作成")
    for _sc, _lab in (("name_resolve.py", "名寄せ表"),
                      ("build_course_turn.py", "コース表")):
        if not run_step(_sc, [PYTHON, os.path.join(BASE_DIR, _sc), "--rebuild"],
                        timeout=900):
            log(f"  [警告] {_lab}の再作成に失敗。前回の表のまま続けます")
            log("         新しい騎手・調教師の成績が欠損になる可能性があります")

    # ── Step 0.9: 壊れたレースクラスを手元のデータで復旧 ──────────────────
    #   2026年6月から結果ページの取得が壊れ、レースクラス欄に条件文が入っていた。
    #   取得側は直したが Step0 は直近2週しか取り直さないので、
    #   過去分はここで埋める。**スクレイピングはしない。**
    #     ① 自分の予想記録(history_marks.csv)から実測を持ってくる
    #     ② 同じ馬が走った他レースのクラスから多数決（確信度0.7以上のみ）
    #   race_data_clean.csv は毎週作り直されるので、毎週かけ直す必要がある。
    log("[Step0.9] レースクラスの復旧")
    if not run_step("recover_class.py",
                    [PYTHON, os.path.join(BASE_DIR, "recover_class.py")],
                    timeout=1800):
        log("  [警告] クラスの復旧に失敗。クラス変化・距離×クラスが欠損します")

    # ── Step 1: 特徴量再生成 ───────────────────────────────────────────────
    log("[Step1] 特徴量再生成")
    run_step(
        "features.py",
        [PYTHON, "-c",
         "import sys; sys.path.insert(0, r'" + BASE_DIR + "'); "
         "from features import build_features; "
         "build_features()"],
        timeout=3600  # 60分
    )

    # ── モデル固定モード（2026-08-09）──────────────────────────────────
    #   FREEZE_MODEL というファイルを置くと、学習をやり直す工程(Step2/3/3.5/3.6)
    #   を飛ばす。予想は今の model.pkl / model_mf.pkl / mf_calibrator.pkl で
    #   出し続ける。
    #
    #   なぜ用意したか: 印の成績や評価の精度を数ヶ月かけて検証したいのに、
    #   毎週モデルが変わると「成績が変わったのはモデル更新のせいか偶然か」を
    #   切り分けられない。学習物を固定すれば、貯まったデータは全て同じモデルの
    #   出力になり、そのまま統計にかけられる。
    #
    #   飛ばすのは学習だけで、Step0(結果の取得)とStep1(特徴量・各種集計の更新)は
    #   続ける。新しいレース結果は貯まり続けるので、検証が終わってから
    #   このファイルを消せば、その時点の全データで学習し直せる。
    frozen = os.path.exists(os.path.join(BASE_DIR, "FREEZE_MODEL"))
    if frozen:
        log("[モデル固定] FREEZE_MODEL があるため Step2/3/3.5/3.6 を飛ばします")
        log("            （予想は今の学習物のまま。結果の蓄積と特徴量更新は続行）")
        STEP_RESULTS.append(("Step2〜3.6 学習（固定モードのため実施せず）", True, 0))

    # ── Step 2: 通常モデル再学習（本番モード=全データ学習 → model.pkl）───────
    #   2026-07-16: timeout 30分→4時間（7/13に30分超過で毎週失敗していたのを修正）。
    #   backtest資産(model_result*.csv)は本番モードでは更新されない（model.py backtestで別途）。
    log("[Step2] モデル再学習（通常モデル・全データ）")
    if not frozen:
      run_step(
        "model.py",
        [PYTHON, os.path.join(BASE_DIR, "model.py")],
        timeout=14400  # 4時間
      )

    # ── Step 3: MFモデル再学習（改善版v2・本番モード=全データ → model_mf.pkl）──
    #   2026-07-16: 旧market_free_model.py→train_mf_v2.py（バギング+LambdaRank版）に切替。
    log("[Step3] MFモデル再学習（train_mf_v2）")
    if not frozen:
      run_step(
        "train_mf_v2.py",
        [PYTHON, os.path.join(BASE_DIR, "train_mf_v2.py")],
        # 2026-08-03に7200秒で足りずタイムアウトした（他の処理と並走していた影響もある）。
        # 翌日は開催がないので余裕を持たせる。
        timeout=21600  # 6時間
      )

    # ── Step 3.5: 確率較正用のOOS出力を更新 ──────────────────────────────
    #   買い判定は「較正済み勝率 × 実オッズ」の期待値で行う（2026-08-04導入）。
    #   MFは正例重み(win 2.0/place3 1.5)と時間重みで学習しているため生の確率は
    #   過大で、2025 OOSでは 勝率 予測11.0%→実際7.2%（1.5倍）だった。
    #   較正しないと期待値も推奨賭け率もずれ、買う馬自体が変わる。
    #   較正器は「正直なOOS出力」から作る必要があり、backtestモード
    #   （≤前年学習/当年検証）でのみ model_mf_result.csv が更新される。
    #   本番モデル(model_mf.pkl)には触らない（出力は model_mf_bt.pkl）。
    log("[Step3.5] 較正用のOOS出力を更新（train_mf_v2 backtest）")
    ok_bt = False if frozen else run_step(
        "train_mf_v2.py backtest",
        [PYTHON, os.path.join(BASE_DIR, "train_mf_v2.py"), "backtest"],
        timeout=21600  # 6時間
    )

    # ── Step 3.6: 確率較正器を作り直す ──────────────────────────────────
    #   モデルが変われば確率の癖も変わるので、毎週作り直す。
    #   失敗しても keiba_predict は較正器が無ければ生の確率で動く（予測は止まらない）
    #   が、その場合は期待値が過大になり買い過ぎるため、警告を残す。
    if ok_bt:
        log("[Step3.6] 確率較正器を再作成")
        if not run_step("build_calibrator.py",
                        [PYTHON, os.path.join(BASE_DIR, "build_calibrator.py")],
                        timeout=1800):
            log("  [警告] 較正器の再作成に失敗。古い較正器のまま動きます")
    elif frozen:
        log("[Step3.6] スキップ（モデル固定モード。較正器も今のまま使います）")
    else:
        log("[Step3.6] スキップ（Step3.5が失敗したため較正器は更新しない）")
        log("  [警告] 較正器が古いままです。期待値がずれる可能性があります")

    # ── Step 3.7: 残差モデルの再学習（2026-08-17追加）─────────────────────
    #   市場のオッズを出発点にして「市場が外している分」だけを学ぶモデル。
    #   従来のMFとは別物で、FREEZE_MODEL の対象外にする。理由は2つ。
    #     ① まだ購入に使っていない（買わずに記録するだけ）。固定する意味が無い
    #     ② 学習を古くすると成績が落ちる。2023年までで固定して2025年を当てると
    #        70%まで落ちた（毎年学習し直せば136%）。こまめな再学習が要る
    #   失敗しても予想は止まらない（残差モデルが無ければ記録しないだけ）。
    # ── Step 3.65: 評価(S/A/B/D)の較正器を作り直す（2026-08-18追加）────────
    #   評価は市場とモデルの2次元ロジスティック（grade_calib.pkl）で決めている。
    #   モデルや特徴量が変われば確率の癖も変わるので、較正器も作り直す必要がある。
    #   週次に入っておらず、古い較正器のまま評価が付き続ける穴だった。
    #   ⚠ bet_cache が要る。無ければ黙って飛ばす（評価は前回の較正器のまま）。
    if os.path.exists(os.path.join(BASE_DIR, "bet_cache_2025.csv")):
        log("[Step3.65] 評価の較正器を再作成")
        if not run_step("build_grade.py",
                        [PYTHON, os.path.join(BASE_DIR, "build_grade.py")],
                        timeout=1800):
            log("  [警告] 評価の較正器の再作成に失敗。前回のまま動きます")
    else:
        log("[Step3.65] スキップ（bet_cache が無いので較正器は作れません）")

    log("[Step3.7] 残差モデル再学習（train_resid・買い判断には未使用）")
    if not run_step("train_resid.py",
                    [PYTHON, os.path.join(BASE_DIR, "train_resid.py")],
                    timeout=7200):
        log("  [警告] 残差モデルの再学習に失敗。前回のモデルのまま記録を続けます")

    # ── Step 3.75: 検証データを作り直す ──────────────────────────────────
    #   ⚠ これが無いと、検証と本番で違うモデルが動く（2026-08-29に発覚）
    #     Step3.7 で model_resid.pkl を学習し直し、Step1系で race_features.csv も
    #     作り直しているのに、**検証データ(resid_kinds_pred.csv)だけが古いまま**
    #     残っていた。実際に8日ぶんずれ、check_resid.py は
    #     「✅ 実装は検証どおり」と出し続けた。選び方は合っていたからである。
    #
    #     結果として何が起きたか
    #       本番の芝の軸は平均8.7番人気、検証は5.9番人気。95%区間の外。
    #       gapの分布も頭数もほぼ同じなのに、選ぶ馬だけが違っていた。
    #     6原因の④「検証と本番で同じものを計算していない」そのもの。
    #
    #   ⚠ 重い（5年ぶんの学習）。失敗しても週次全体は止めない。
    #     止めると本番の予想まで巻き添えになるため。
    log("[Step3.75] 検証データ(resid_kinds_pred.csv)を現行モデルで作り直す")
    if not run_step("resid_kinds.py",
                    [PYTHON, os.path.join(BASE_DIR, "resid_kinds.py")],
                    timeout=10800):
        log("  [警告] 検証データを作り直せませんでした。")
        log("         **check_resid.py の数字は本番の成績を表しません。**")
        log("         手で python resid_kinds.py を実行してください")

    # ── Step 3.8: 残差モデルの前向き検証レポート ─────────────────────────
    #   買い率が想定(12.6%)から外れていないか、実測の回収率がどうかを毎週見る。
    #   的中100本を超えるまでは数字を信用しない（バックテストでも年ごとの
    #   的中は28〜67本で、その本数だとROIは倍半分に振れる）。
    log("[Step3.8] 残差モデルの記録を集計")
    run_step("paper_report.py",
             [PYTHON, os.path.join(BASE_DIR, "paper_report.py")], timeout=600)

    # ── Step 4: result_tracker 更新 ───────────────────────────────────────
    log("[Step4] レース結果照合")
    run_step(
        "result_tracker update",
        [PYTHON, os.path.join(BASE_DIR, "result_tracker.py"), "update"],
        # 2026-08-10: 300秒では未処理280件を捌けず毎週打ち切られていた。
        # 1件あたり約1.5秒（取得＋待機）なので、溜まった分を消化できる長さにする。
        timeout=2400
    )
    run_step(
        "result_tracker summary",
        [PYTHON, os.path.join(BASE_DIR, "result_tracker.py"), "summary"],
        timeout=120
    )

    # ── Step 4.5: 精度サマリー自動生成 ───────────────────────────────────
    log("[Step4.5] 精度サマリー生成")
    try:
        import pandas as pd
        import numpy as np

        rec_path = os.path.join(BASE_DIR, "prediction_record_v2.csv")
        df = pd.read_csv(rec_path)
        confirmed = df[df["hit"].notna()].copy()
        confirmed["honmei_actual"] = pd.to_numeric(confirmed["honmei_actual"], errors="coerce")
        confirmed["honmei_ninki"]  = pd.to_numeric(confirmed["honmei_ninki"], errors="coerce")
        confirmed["honmei_ev"]     = pd.to_numeric(confirmed["honmei_ev"], errors="coerce")

        n = len(confirmed)
        overall_win  = (confirmed["honmei_actual"] == 1).mean() * 100
        overall_fuku = (confirmed["honmei_actual"] <= 3).mean() * 100

        by_jyo = confirmed.groupby("jyo").agg(
            件数=("honmei_actual", "count"),
            勝率=("honmei_actual", lambda x: (x==1).mean()*100),
        ).reset_index()

        issues = []
        if overall_win < 25:
            issues.append(f"◎勝率低下({overall_win:.1f}%) → コース形態/血統特徴量の効果確認")
        weak = by_jyo[by_jyo["勝率"] < 20]
        if not weak.empty:
            issues.append(f"弱点競馬場: {', '.join(weak['jyo'].tolist())} → 直線長_m・坂あり要チェック")
        nb1_pct = (confirmed["honmei_ninki"] == 1).mean() * 100
        if nb1_pct > 70:
            issues.append(f"◎の{nb1_pct:.0f}%が1番人気 → MF識別力不足、B-1血統データ追加を検討")
        ev_pos = confirmed[confirmed["honmei_ev"] >= 0]
        ev_neg = confirmed[confirmed["honmei_ev"] < 0]
        if len(ev_pos) > 5 and len(ev_neg) > 5:
            wp  = (ev_pos["honmei_actual"] == 1).mean() * 100
            wn  = (ev_neg["honmei_actual"] == 1).mean() * 100
            if wp < wn:
                issues.append(f"EV≥0の勝率({wp:.1f}%) < EV<0({wn:.1f}%) → EVフィルター閾値再検討")

        summary_path = os.path.join(BASE_DIR, "accuracy_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"=== 精度分析サマリー {datetime.now().strftime('%Y/%m/%d')} ===\n\n")
            f.write(f"◎勝率: {overall_win:.1f}%  複勝率: {overall_fuku:.1f}%  照合件数: {n}件\n\n")
            f.write("【競馬場別】\n")
            for _, row in by_jyo.iterrows():
                f.write(f"  {row['jyo']}: {row['勝率']:.1f}% ({row['件数']}件)\n")
            f.write("\n【月曜改修ポイント】\n")
            for issue in (issues or ["現時点で明確な弱点なし"]):
                f.write(f"- {issue}\n")
        log(f"  精度サマリー保存: {summary_path}")
    except Exception as e:
        log(f"  精度サマリー生成エラー: {e}")

    # ── Step 5: GitHubプッシュ ────────────────────────────────────────────
    log("[Step5] GitHubプッシュ")
    try:
        push_files = [
            "prediction_record_v2.csv",
            "model_result.csv",
            "weekly_update_log.txt",
            "accuracy_log.csv",
            "accuracy_summary.txt",
            # 確率較正器（2026-08-05追加・7KB）。買い判定が較正済み確率に
            # 依存するようになったため、これが古いと期待値がずれる。
            "mf_calibrator.pkl",
        ]
        for f in push_files:
            fpath = os.path.join(BASE_DIR, f)
            if os.path.exists(fpath):
                subprocess.run(["git", "add", fpath], cwd=BASE_DIR, capture_output=True)

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=BASE_DIR, capture_output=True, text=True
        )
        if status.stdout.strip():
            date_str = datetime.now().strftime("%Y/%m/%d")
            subprocess.run(
                ["git", "commit", "-m", f"週次自動更新 {date_str}"],
                cwd=BASE_DIR, capture_output=True
            )
            subprocess.run(["git", "push"], cwd=BASE_DIR, capture_output=True)
            log("  GitHubプッシュ完了")
        else:
            log("  変更なし・スキップ")
    except Exception as e:
        log(f"  Gitエラー: {e}")

    elapsed = datetime.now() - start
    log("=" * 50)
    log(f"週次自動更新 完了  所要時間: {elapsed}")
    log("=" * 50)
    _notify(elapsed)


def _notify(elapsed):
    """結果をメールで知らせる（2026-08-09追加）。

    これまでは weekly_update_log.txt に書くだけだったので、8/3にStep3が
    時間切れでMFモデルが更新されないまま終わっていたことに1週間気づけなかった。
    無人運用では、失敗が届かないことが一番こわい。
    """
    ng = [l for l, ok, _ in STEP_RESULTS if not ok]
    head = "✅週次更新 完了" if not ng else f"⚠週次更新 {len(ng)}件が失敗"
    lines = [f"所要時間 {elapsed}", ""]
    for l, ok, sec in STEP_RESULTS:
        lines.append(f"  {'○' if ok else '×'} {l}  {sec // 60}分")
    lines.append("")
    # モデル固定モードでは学習物が古いのは意図どおりなので、警告にしない。
    # 「6日前のまま」と出ると、固定していることを忘れた将来の自分が
    # 障害と誤解する（2026-08-10の実メールで実際に紛らわしかった）。
    frozen = os.path.exists(os.path.join(BASE_DIR, "FREEZE_MODEL"))
    lines.append("モデルファイルの更新日時:"
                 + ("（モデル固定中。学習物が古いのは意図どおり）" if frozen else ""))
    for f in ("model.pkl", "model_mf.pkl", "mf_calibrator.pkl", "race_features.csv"):
        fp = os.path.join(BASE_DIR, f)
        if os.path.exists(fp):
            m = datetime.fromtimestamp(os.path.getmtime(fp))
            age = (datetime.now() - m).days
            if f == "race_features.csv":
                note = "  ← 今日更新" if age == 0 else f"  ← {age}日前（更新されていない）"
            elif frozen:
                note = "  （固定中）"
            else:
                note = f"  ← {age}日前のまま" if age >= 6 else ""
            lines.append(f"  {f}: {m:%m/%d %H:%M}{note}")
        else:
            lines.append(f"  {f}: 見つからない")
    # ── データの欠けを毎週チェックする（2026-08-18追加）────────────────────
    #   血統マスタが1か月半止まっていたのに誰も気づかなかった。工程が「成功」でも
    #   中身が欠けていることはあるので、結果そのものを見る。
    lines.append("")
    lines.append("データの欠け:")
    try:
        import pandas as _pd
        _hm = _pd.read_csv(os.path.join(BASE_DIR, "horse_master.csv"), dtype=str)
        _hm["horse_id"] = (_hm["horse_id"].astype(str)
                           .str.replace(".0", "", regex=False).str.strip())
        _rc = _pd.read_csv(os.path.join(BASE_DIR, "race_data_clean.csv"),
                           usecols=["race_id", "horse_id"], dtype=str, low_memory=False)
        _rc["horse_id"] = (_rc["horse_id"].astype(str)
                           .str.replace(".0", "", regex=False).str.strip())
        _rc["年"] = _rc["race_id"].str.replace(r"\.0$", "", regex=True).str[:4]
        _have = set(_hm["horse_id"])
        _cur = str(datetime.now().year)
        _now = _rc[_rc["年"] == _cur]
        _miss = _now[~_now["horse_id"].isin(_have)]["horse_id"].nunique()
        _tot = _now["horse_id"].nunique()
        _pct = _miss / _tot * 100 if _tot else 0
        _mark = "" if _pct < 2 else ("  ← 要確認" if _pct < 10 else "  ← 異常")
        lines.append(f"  血統が取れていない馬({_cur}年): {_miss}/{_tot}頭"
                     f"（{_pct:.1f}%）{_mark}")
    except Exception as e:
        lines.append(f"  血統チェックに失敗: {type(e).__name__}")
    for _f, _lab in (("race_features.csv", "特徴量"),
                     ("sire_stats_father.csv", "種牡馬成績"),
                     ("model_resid.pkl", "残差モデル")):
        _p = os.path.join(BASE_DIR, _f)
        if os.path.exists(_p):
            _a = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(_p))).days
            lines.append(f"  {_lab}の更新: {_a}日前"
                         + ("  ← 止まっている" if _a > 10 else ""))
        else:
            lines.append(f"  {_lab}: 見つからない  ← 異常")

    if ng:
        lines += ["", "失敗したもの:"] + [f"  {l}" for l in ng]
        lines += ["", "詳しくは weekly_update_log.txt を見てください。"]
    body = chr(10).join(lines)
    log(body)
    try:
        import smtplib
        from email.mime.text import MIMEText
        from dotenv import load_dotenv
        load_dotenv(os.path.join(BASE_DIR, ".env"))
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = f"{head}（{datetime.now():%m/%d}）"
        msg["From"] = os.environ["GMAIL_ADDRESS"]
        msg["To"] = os.environ["TO_ADDRESS"]
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(os.environ["GMAIL_ADDRESS"], os.environ["GMAIL_APP_PASS"])
            srv.send_message(msg)
        log("  完了メールを送信しました")
    except Exception as e:
        log(f"  完了メールの送信に失敗: {e}")


if __name__ == "__main__":
    main()
