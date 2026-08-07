# keiba_ai 環境セットアップ・トラブル対応ガイド

## これまで起きていた問題

`ModuleNotFoundError` が繰り返し発生していたのは、PCに複数の Python が
混在しており（WindowsApps の `python3.11.exe`、`pythoncore-3.14-64` など）、
その時々でどれが実行されるかが不安定だったためです。
ライブラリをインストールしても「別の Python」に入っていなければ
見つからない、という状態が起きていました。

## 今後の運用ルール（これだけ守れば迷わない）

### 1. 最初の1回だけ: セットアップスクリプトを実行

```powershell
cd "C:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai"
.\setup_venv.ps1
```

これで `venv` フォルダ（プロジェクト専用の Python 環境）が作られ、
`requirements.txt` に書かれた全ライブラリが一括インストールされます。
このフォルダはプロジェクトの中にあるので、PC がスリープ/シャットダウンしても
消えることはありません。

### 2. 以後、スクリプトを実行する前に必ずこの2行

```powershell
cd "C:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai"
.\venv\Scripts\Activate.ps1
```

プロンプトの先頭に `(venv)` と表示されればOKです。
この状態で `python xxx.py` を実行すれば、常に同じライブラリ一式が使われます。

### 3. 新しいライブラリが必要になったら

コード側で新しい `import` を追加したときは、`requirements.txt` にも
その名前を1行追加してから、以下を実行してください（venv を有効化した状態で）。

```powershell
pip install -r requirements.txt
```

### 4. 「ModuleNotFoundError」が出たときの確認手順

1. `(venv)` がプロンプトに出ているか確認する（出ていなければ手順2を実行）
2. それでも出るなら、そのライブラリ名が `requirements.txt` に入っているか確認する
3. 入っていなければ追記して `pip install -r requirements.txt` を再実行する

### 5. タスクスケジューラ（weekly_update.py など）の実行ファイルパスも要更新

`weekly_update.py` の中の `PYTHON` 変数は、現在
`C:/Users/別府飛河/AppData/Local/Microsoft/WindowsApps/python3.11.exe` を
指していますが、今後は venv 内の Python に統一するのがおすすめです。

```python
PYTHON = r"c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai\venv\Scripts\python.exe"
```

こう変更しておけば、タスクスケジューラ実行時も含めて
「どの Python が動くか分からない」問題が根本的になくなります。
