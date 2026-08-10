# 運用ガイド（やること・困ったときの対処）

システムの中身は [SYSTEM.md](SYSTEM.md) を見てください。
こちらは「何をすればいいか」「壊れたらどう直すか」だけを書いています。

最終更新: 2026-08-10

---

## 1. 平常時にやること

### 週1回（火曜の朝）

**週次更新メールを見るだけ。** これだけです。

| 件名 | 意味 | 対応 |
|---|---|---|
| `✅週次更新 完了` | 正常 | 何もしなくてよい |
| `⚠週次更新 N件が失敗` | どこかが失敗 | 本文にどのStepか書いてある → 4章へ |
| **メールが来ない** | **最も危険な兆候** | タスクごと止まっている可能性 → 4章へ |

メール本文には各Stepの成否・所要時間と、モデルファイルの更新日時が載ります。
**モデル固定中は「（固定中）」と出るのが正常**です。`race_features.csv` だけは
「← 今日更新」と出るべきで、「N日前（更新されていない）」なら異常です。

### 月1回

`history_marks.csv` の行数が増えているか見てください。
**開催日1日あたり約486行**（36レース × 約13.5頭）増えます。

```powershell
cd "c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai"
(Get-Content history_marks.csv | Measure-Object -Line).Lines
```

増えていなければ蓄積が止まっています → 4章「蓄積が止まった」へ。

### 見なくていいもの

見張り番が20分おきに自動で見ています。**異常があればメールが来ます**。
自力で直せた場合は通報しないので、**メールが来ない＝正常**です。

---

## 2. 自動で動いているもの

| 時刻 | 内容 | 実体 |
|---|---|---|
| ログオン時 | ダッシュボード（Flask + ngrok） | スタートアップ `競馬AIダッシュボード.lnk` |
| 6:55 | 常駐スケジューラ起動 | タスク `競馬AI自動予想` |
| 7:00 | 全レース予想 → GitHubへpush | 常駐 |
| 7:00直後 | `keiba_auto.py` 起動（7分前ジョブ用） | 常駐 |
| 7:05 | 個別レース登録・発走時刻を保存 | 常駐 |
| 各レース40分前 | 再予想 ＋ 前レースの結果取得 | 常駐 |
| 各レース7分前 | 直前オッズで買い目確定 ＋ メール | keiba_auto |
| 17:00〜20:40 | 結果の後片付け | 常駐 |
| 21:00 | 結果照合 | タスク `keiba_analyze_accuracy` |
| 21:10 | **日次アーカイブ**（`history_marks.csv`） | 常駐 |
| 22:30 | 常駐終了（翌朝6:55に再起動） | 常駐 |
| 20分おき | 見張り番 | タスク `競馬AI見張り番` |
| 火曜 8:00 | 週次更新（開催日なら自動見送り） | タスク `競馬AI週次更新` |

**非開催日は、朝の予想も日次アーカイブも自動でスキップされます。**

---

## 3. 見るべきファイル

| ファイル | 何が分かるか |
|---|---|
| `auto_predict_heartbeat.txt` | スケジューラが生きているか（**中身が現在時刻に追随していれば正常**） |
| `history_marks.csv` | 蓄積の本体。開催日ごとに約486行増える |
| `today_results.log` | 結果取得の記録 |
| `watchdog.log` | 見張り番の判定履歴 |
| `weekly_update_log.txt` | 週次更新の詳細 |
| `keiba_auto_run.err` | 7分前ジョブの例外 |
| `flask_run.err` | ダッシュボードのエラー |

---

## 4. 困ったときの対処

### 症状：週次更新のメールが来ない

```powershell
# タスクの状態を見る
Get-ScheduledTask -TaskName "競馬AI週次更新" | Get-ScheduledTaskInfo
```

`NextRunTime` が空なら、トリガーが切れています。再登録してください（7章）。
`LastTaskResult` が0以外なら失敗しています。`weekly_update_log.txt` の末尾を見ます。

### 症状：ダッシュボードが開かない

```powershell
cd "c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai"
python start_dashboard.py
```

Flask と ngrok の両方を起動し直します（二重起動ガードがあるので何度実行しても安全）。
Flaskだけ直したいときは `python flask_app.py`（ngrokは動いたままなのでURLは変わりません）。

### 症状：予想が更新されない／心拍が止まっている

まず心拍を見ます。

```powershell
Get-Content auto_predict_heartbeat.txt
```

現在時刻から**8分以上遅れていたら、生きたまま固まっています**。
見張り番が20分以内に自動で再起動しますが、急ぐなら手動で。

```powershell
schtasks /End /TN "競馬AI自動予想"
schtasks /Run /TN "競馬AI自動予想"
```

> ⚠️ **`Stop-Process` では止まりません。** 管理者権限のタスクなので、
> 必ず `schtasks` かタスクスケジューラ経由で止めてください。

### 症状：7分前の直前更新が動かない（`直前更新` のpushが無い）

```powershell
Get-Content keiba_auto_run.err -Tail 20
```

例外が出ていればそれが原因です。過去には `import schedule` が
コメントアウトされていて起動15秒で落ちていたことがあります。

見張り番が15分おきに自動で起動し直しますが、手動なら常駐を再起動してください。

### 症状：蓄積が止まった（`history_marks.csv` が増えない）

原因は上流から順に3つ考えられます。

1. **予想が動いていない** → `today_predictions.csv` の更新日時を見る
2. **結果が取れていない** → `today_results.log` を見る。
   netkeibaのHTML変更やブロックの可能性
3. **アーカイブだけ失敗** → 21:10前後の常駐ログ

その日のうちなら手動で積めます。

```powershell
python archive_daily.py            # 当日分
python archive_daily.py 2026-08-09 # 日付を指定して過去分
```

> 非開催日には自動でスキップされます（前回の分を積み直さないため）。

### 症状：IPブロックされた（CloudFrontで400など）

**すぐに常駐を止めてください。** 取得を続けると悪化します。

```powershell
schtasks /End /TN "競馬AI自動予想"
```

数時間〜1日空けてから再開します。
現在の設計では1日の取得回数は最大でもレース数と同じ（36回程度）なので、
**通常運転でブロックされることはまず無い**はずです。起きたら設計を疑ってください。

### 症状：ディスクが足りない

閾値2GBを切ると自動でメール警告が来て、予想は中止されます。

2026-08-10に約7.3GBを整理済み（重複したモデルのバックアップを削除）。
それでも足りない場合、消してよいのは次のものです。

| 消せるもの | 理由 |
|---|---|
| `model_mf_bt.pkl`（もし再生成されていたら） | バックテスト用。出力の `model_mf_result.csv` さえあればよい |
| `model_mf_backup.pkl`（同上） | 学習をやり直せば作り直される |
| `race_features.csv` | 週次のStep1で作り直される（1.3GB） |

> ⚠️ **`model_mf_parts/` と `model_mf_parts.verified_0801` は消さないこと。**
> 前者は本番が読むモデル、後者は**唯一の復旧手段**です。

### 症状：モデルが壊れた（予想が出ない・読み込みエラー）

本番は `model_mf_parts/` を読みます。まず完全性を確認してください。

```powershell
python -c "import pickle,os; m=pickle.load(open('model_mf_parts/meta.pkl','rb')); fs=os.listdir('model_mf_parts'); print({t: (len([f for f in fs if f.startswith(t+'_')]), n) for t,n in m['counts'].items()})"
```

`win`/`place2`/`place3` がそれぞれ **(10, 10)** ならば完全です。
数が合わない、または `meta.pkl` が読めない場合は保険から戻します。

```powershell
Rename-Item model_mf_parts model_mf_parts.broken
Copy-Item -Recurse model_mf_parts.verified_0801 model_mf_parts
```

`model_mf_parts/` ごと失われた場合は、`model_mf.pkl`（フォールバック）が
あれば自動的にそちらが使われるので、予想は止まりません。

### 症状：PCを再起動した

- **タスク**（自動予想・見張り番・週次更新・結果照合）は**自動で復帰**します
- **ダッシュボード**はログオン時のスタートアップで起動します。
  ログオンしていないと立ち上がりません（蓄積には影響しません）

再起動後に確認するなら、心拍が動いているかだけ見れば十分です。

```powershell
Get-Content auto_predict_heartbeat.txt
```

---

## 5. やってはいけないこと

| やってはいけない | 理由 |
|---|---|
| **`Stop-Process` で常駐を止める** | 管理者権限のタスクなので止まらない。`schtasks` を使う |
| **大量スクレイピングを走らせる** | 2026-07-27にCloudFrontで400を食らっている |
| **買い方の設定を思いつきで変える** | 100通り以上検証して現行が最良。変えると蓄積の意味が失われる |
| **`FREEZE_MODEL` を軽い気持ちで消す** | 消すと次の火曜に再学習が走り、蓄積データの前提（同一モデル）が崩れる |
| **数字を再現せずに公開する** | 検証は `bet_cache_*.csv` ＋ `jv_payouts.csv` で行う。別の入力から自前計算すると再現しない |

---

## 6. 半年後にやること（優先順）

### まず確かめること

1. **検証と実運用のズレ**
   `history_marks.csv` の `オッズ変化率` から、7分前に買った場合の期待値が
   確定オッズ基準と何%違うかを測る。**これが分からないと他の全ての数字の意味が定まらない。**
2. **買い方の候補3案が2026年で再現するか**（[SYSTEM.md](SYSTEM.md) 6章）
   最有望は「馬単の相手を人気2位以内」（3年通算131.5%・下限97.0・**検体が減らない**）
3. **▲と△の勝率逆転**が本物か（294レース必要。約1ヶ月分）
4. 評価グレードの実運用での較正（300レース必要）

### 分析の始め方

```python
import pandas as pd
d = pd.read_csv("history_marks.csv", dtype={"race_id": str})

# 印別
for m in "◎○▲△×":
    s = d[d.推奨ランク == m]
    print(m, round(s["1着"].mean()*100, 1), round(s["3着内"].mean()*100, 1))

# モデルが変わった時期で分けたいとき
d.groupby("MF版")["3着内"].mean()
```

### 検証が終わったら

1. `FREEZE_MODEL` を削除 → 次の火曜に全データで学習し直す
2. 通常モデル（`model.pkl`）の除去を検討（フォールバックが不要と確認できた場合）
3. `year_sweep.py` を再実行（テスト年が2026まで増えるので検体1.5倍）

### 新しいデータを足すなら

**唯一まだ試していない質的に異なる情報が「直前の資金の動き」**です。
過去データの加工は10回試して全て失敗しています（[SYSTEM.md](SYSTEM.md) 6章）。

1. オッズ変動を特徴量にする（数千レース必要。2026-07-23から蓄積中）
2. それで効果が出れば、**票数**（JV-Link `H1`）に進む。オッズは票数の比率にすぎず、
   票数のほうが「どれだけの金額が、どの馬に、いつ入ったか」という生の情報

---

## 7. 復旧コマンド集

```powershell
cd "c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai"

# 常駐スケジューラの再起動
schtasks /End /TN "競馬AI自動予想" ; schtasks /Run /TN "競馬AI自動予想"

# ダッシュボードの起動（Flask + ngrok + URLメール）
python start_dashboard.py

# 見張り番を手で動かす（--dry なら通報せず判定だけ）
python watchdog.py --dry

# 週次更新を手で動かす（開催日なら自動で見送る）
python weekly_update.py

# 当日結果を手で取る
python today_results.py sweep

# 蓄積を手で積む
python archive_daily.py 2026-08-09

# note用のダイジェストを送らずに表示
python note_digest.py --print
```

### タスクの再登録（トリガーが切れた場合）

```powershell
# 見張り番（毎日6:50から20分おき・18時間）
$py = "C:/Users/別府飛河/AppData/Local/Microsoft/WindowsApps/python3.11.exe"
$dir = "c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai"
$a = New-ScheduledTaskAction -Execute $py -Argument "$dir\watchdog.py" -WorkingDirectory $dir
$t = New-ScheduledTaskTrigger -Daily -At 6:50am
$t.Repetition = (New-ScheduledTaskTrigger -Once -At 6:50am `
    -RepetitionInterval (New-TimeSpan -Minutes 20) `
    -RepetitionDuration (New-TimeSpan -Hours 18)).Repetition
$s = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "競馬AI見張り番" -Action $a -Trigger $t -Settings $s -Force
```

> `競馬AI週次更新` と `競馬AI自動予想` は管理者権限で作られているため、
> 変更には**管理者として実行したPowerShell**が必要です。

---

## 8. 覚えておくべき教訓

これまでの障害は、ほぼ全てこの4つのどれかでした（[SYSTEM.md](SYSTEM.md) 12章に詳細）。

1. **タスクスケジューラ起動のプロセスは標準出力が消える。**
   ログをファイルに書かないと死因が永久に分からない
2. **「プロセスが存在する＝正常」ではない。**
   生きたまま固まる。心拍のような能動的な証跡が要る
3. **状況証拠だけで原因を決めない。**
   メモリ不足と診断したが実際は `NameError` だった
4. **毎日動く処理は「今日のデータか」を必ず確認する。**
   `today_*.csv` は開催日にしか更新されない。ガードが無いと前回の分を積み直す
