# -*- coding: utf-8 -*-
"""model_mf の分割保存/逐次読込。
2.5GBの単一pickleを一括loadするとデシリアライズのピークRAMが空きを食い潰して
クラッシュする(2026-07-26発生: 空きRAM1GB時に読込不能で予想停止)。
モデルを1個ずつ別ファイルに保存し逐次読み込むことで、読込ピークを
「積み上がり分＋モデル1個」に抑える。予測値は完全に不変(モデル自体は同一)。

- save_mf_split(save, base_dir): model_mf_parts/ に meta + 各モデルを書き出す
- load_mf(base_dir): parts があれば逐次読込、無ければ従来 model_mf.pkl にフォールバック
"""
import os
import gc
import pickle
import shutil
import time

PARTS_DIR = "model_mf_parts"
TARGETS = ["win", "place2", "place3"]


def save_mf_split(save: dict, base_dir: str = "."):
    """train_mf_v2 の save dict を分割保存。tmpに書き切ってから入れ替え(中途半端な状態を防ぐ)。"""
    final = os.path.join(base_dir, PARTS_DIR)
    tmp = final + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    # 2026-08-14: 距離2モデル構成に対応。長距離(全特徴)と短距離(騎手厩舎なし)の
    #   両方を保存する。dist_split が無い旧形式の save でもそのまま動く。
    meta = {"format": "multi_v2_split", "use_cols": save["use_cols"], "counts": {},
            "dist_split": save.get("dist_split"),
            "use_cols_short": save.get("use_cols_short"), "counts_short": {}}
    for t in TARGETS:
        models = save[t]["models"]
        meta["counts"][t] = len(models)
        for i, m in enumerate(models):
            with open(os.path.join(tmp, f"{t}_{i:02d}.pkl"), "wb") as f:
                pickle.dump(m, f, protocol=pickle.HIGHEST_PROTOCOL)
        short = save[t].get("models_short") or []
        meta["counts_short"][t] = len(short)
        for i, m in enumerate(short):
            with open(os.path.join(tmp, f"{t}_short_{i:02d}.pkl"), "wb") as f:
                pickle.dump(m, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(tmp, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    # 旧ディレクトリの削除はOneDriveがロックして PermissionError になることがある
    # (2026-07-29に発生。中身は消えたがディレクトリだけ残り、tmpからの入れ替えに失敗した)。
    # リトライしても消えない場合は、tmpを消さずに残して手動復旧できるようにする。
    if os.path.exists(final):
        for i in range(5):
            try:
                shutil.rmtree(final)
                break
            except PermissionError:
                time.sleep(2)
        else:
            try:
                os.rename(final, final + f".old{int(time.time())}")
            except OSError as e:
                raise RuntimeError(
                    f"旧{final}を退けられません({e})。{tmp} に新しい分割モデルが"
                    f"揃っているので、手動で {final} を消して tmp をリネームしてください") from e
    os.replace(tmp, final)
    return final


def load_mf(base_dir: str = "."):
    """MFモデルを読み込み、従来のmodel_mf.pklと同じ形のdictを返す。
    model_mf_parts/ があれば逐次読込(低ピークRAM)、無ければ従来pklにフォールバック。
    失敗時は例外を投げる(リトライ/警告は呼び出し側)。"""
    parts = os.path.join(base_dir, PARTS_DIR)
    meta_path = os.path.join(parts, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        out = {"format": "multi_v1", "use_cols": meta["use_cols"],
               "dist_split": meta.get("dist_split"),
               "use_cols_short": meta.get("use_cols_short")}
        cs = meta.get("counts_short") or {}
        for t in TARGETS:
            models = []
            for i in range(meta["counts"][t]):
                with open(os.path.join(parts, f"{t}_{i:02d}.pkl"), "rb") as f:
                    models.append(pickle.load(f))
                gc.collect()  # 1個ごとに一時バッファを回収して読込ピークを抑える
            short = []
            for i in range(cs.get(t, 0)):
                with open(os.path.join(parts, f"{t}_short_{i:02d}.pkl"), "rb") as f:
                    short.append(pickle.load(f))
                gc.collect()
            out[t] = {"models": models, "use_cols": meta["use_cols"],
                      "models_short": short,
                      "use_cols_short": meta.get("use_cols_short"),
                      "dist_split": meta.get("dist_split")}
        out["models"] = out["win"]["models"]  # 旧コード互換(win別名)
        return out
    # フォールバック: 従来の単一pickle(一括load・高ピーク)
    with open(os.path.join(base_dir, "model_mf.pkl"), "rb") as f:
        return pickle.load(f)


def exists(base_dir: str = ".") -> bool:
    """読めるMFモデル(分割 or 従来pkl)が存在するか。"""
    return (os.path.exists(os.path.join(base_dir, PARTS_DIR, "meta.pkl"))
            or os.path.exists(os.path.join(base_dir, "model_mf.pkl")))
