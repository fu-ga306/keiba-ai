# -*- coding: utf-8 -*-
"""OOS予測＋較正＋市場確率を一度だけ計算してキャッシュする。

bet_model2.py は4時間半かかった。原因はHarvilleの三重ループ。
式を整理すると O(n^2) に落ちる:
  P(2着)_i = q_i * ( Σ_j q_j/(1-q_j) - q_i/(1-q_i) )
  P(3着)_i = q_i * ( S - M[i,:].sum() - M[:,i].sum() )
      ただし M[j,k] = q_j*q_k/((1-q_j)*(1-q_j-q_k)) (j≠k), S = ΣΣ M
これで秒単位になり、閾値の探索を何度でも回せる。
出力: bet_cache_2021〜2025.csv（各年、前年のOOSで較正）
  2026-08-05: 3年→5年に拡張。CI下限97.9が唯一の制約で、これは年数でしか埋まらない。
  単勝はrace_features(2019〜)だけで検証でき追加データが要らない。
  ※2020のOOSは学習が2019のみと薄いが、2021の較正にしか使わない。
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from market_free_model import FEATURE_COLS_MF
def log(m): print(m, flush=True)

bnm=pd.read_csv("race_data_clean.csv",usecols=["race_id","馬名","馬番"],dtype=str,low_memory=False)
bnm["race_id"]=bnm["race_id"].str.replace(r"\.0$","",regex=True)
bnm["bn"]=bnm["馬番"].str.extract(r"(\d+)")[0].astype(float).astype(int).astype(str).str.zfill(2)
bnm=bnm.drop_duplicates(["race_id","馬名"])[["race_id","馬名","bn"]]

log("読み込み中...")
head=pd.read_csv("race_features.csv",nrows=1)
BASE=[c for c in FEATURE_COLS_MF if c in head.columns]
use=list(dict.fromkeys(["race_id","馬名","着順_num","人気","単勝オッズ"]+BASE))
D=pd.read_csv("race_features.csv",usecols=use,dtype={"race_id":str},low_memory=False)
D["race_id"]=D["race_id"].astype(str).str.replace(r"\.0$","",regex=True)
D["年"]=D["race_id"].str[:4].astype(int)
D=D.merge(bnm,on=["race_id","馬名"],how="left")
D["着"]=pd.to_numeric(D["着順_num"],errors="coerce")
D["odds"]=pd.to_numeric(D["単勝オッズ"],errors="coerce")
D["人気"]=pd.to_numeric(D["人気"],errors="coerce")
D=D[D["着"].notna()&D["odds"].notna()&(D["odds"]>0)&D["bn"].notna()]
D["win"]=(D["着"]==1).astype(int)
D["top2"]=(D["着"]<=2).astype(int)
D["top3"]=(D["着"]<=3).astype(int)
D["頭数"]=D["race_id"].map(D.groupby("race_id").size())
D=D[D["頭数"]>=8]

P=dict(objective="binary",metric="binary_logloss",learning_rate=0.03,num_leaves=63,
       min_data_in_leaf=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,
       verbose=-1,seed=42)

# 距離分割（2026-08-16追加）。本番(train_mf_v2.py)は 1900m を境に長距離/短距離で
# 別モデルを使う。ここを合わせていなかったため、検証と本番で別のモデルを見ていた。
#   実害: 1番人気のMF複勝順位が4位以下になる割合が、検証10.9%に対し本番31%。
#         荒れR方式の対象レースそのものがズレていた（2026-08-16に発見）。
# ⚠ 本番のモデル定義を変えたら、ここも必ず合わせること。
try:
    from market_free_model import MF_DIST_SPLIT, FEATURE_COLS_MF_SHORT
    _SHORT=[c for c in FEATURE_COLS_MF_SHORT if c in head.columns]
    _SPLIT=True
except Exception:
    MF_DIST_SPLIT, _SHORT, _SPLIT = None, None, False
if _SPLIT and "距離" not in D.columns:
    _d=pd.read_csv("race_features.csv",usecols=["race_id","馬名","距離"],
                   dtype={"race_id":str},low_memory=False)
    _d["race_id"]=_d["race_id"].astype(str).str.replace(r"\.0$","",regex=True)
    D=D.merge(_d.drop_duplicates(["race_id","馬名"]),on=["race_id","馬名"],how="left")
if _SPLIT:
    D["_long"]=pd.to_numeric(D["距離"],errors="coerce")>=MF_DIST_SPLIT
    log(f"距離分割 {MF_DIST_SPLIT}m: 長{D['_long'].sum():,}頭 / 短{(~D['_long']).sum():,}頭"
        f" (特徴量 長{len(BASE)} 短{len(_SHORT)})")
else:
    log("⚠ 距離分割なし（market_free_model に定義が無い）。本番と一致しない可能性あり")

def oos(TY):
    tr,te=D[D["年"]<TY],D[D["年"]==TY].copy()
    ymin,ymax=tr["年"].min(),tr["年"].max()
    w=(1.0+(tr["年"]-ymin)/max(ymax-ymin,1)).values
    if not _SPLIT:
        for tgt,pw in (("win",2.0),("top2",1.7),("top3",1.5)):
            m=lgb.train(P,lgb.Dataset(tr[BASE],tr[tgt],weight=w*np.where(tr[tgt]==1,pw,1.0)),
                        num_boost_round=800)
            te[f"p_{tgt}"]=m.predict(te[BASE])
        log(f"  {TY} OOS done")
        return te
    # 本番と同じく、長距離/短距離で別のモデル・別の特徴量集合を使う
    for tgt in ("win","top2","top3"):
        te[f"p_{tgt}"]=np.nan
    for is_long,cols in ((True,BASE),(False,_SHORT)):
        trm,tem=tr["_long"]==is_long, te["_long"]==is_long
        if trm.sum()<500 or tem.sum()==0:
            continue
        wm=w[trm.values]
        for tgt,pw in (("win",2.0),("top2",1.7),("top3",1.5)):
            m=lgb.train(P,lgb.Dataset(tr.loc[trm,cols],tr.loc[trm,tgt],
                                      weight=wm*np.where(tr.loc[trm,tgt]==1,pw,1.0)),
                        num_boost_round=800)
            te.loc[tem,f"p_{tgt}"]=m.predict(te.loc[tem,cols])
    log(f"  {TY} OOS done（長{(te['_long']).sum():,} / 短{(~te['_long']).sum():,}）")
    return te

log("OOS予測（6年）...")
Y={y:oos(y) for y in (2020,2021,2022,2023,2024,2025)}

def harville(d):
    """O(n^2)で複勝圏の市場確率を出す。"""
    inv=1.0/d["odds"]
    d["q_win"]=inv/inv.groupby(d["race_id"]).transform("sum")
    out=np.zeros(len(d))
    for _,idx in d.groupby("race_id",sort=False).indices.items():
        q=np.clip(d["q_win"].values[idx],1e-9,1-1e-9); n=len(q)
        r=q/(1-q)
        p2=q*(r.sum()-r)
        den=1-q[:,None]-q[None,:]
        M=(q[:,None]*q[None,:])/((1-q)[:,None]*np.clip(den,1e-9,None))
        np.fill_diagonal(M,0.0)
        S=M.sum()
        p3=q*(S-M.sum(1)-M.sum(0))
        out[idx]=np.clip(q+p2+p3,1e-6,1.0)
    d["q_top3"]=out
    return d

log("市場確率（Harville・高速版）...")
for y in Y: Y[y]=harville(Y[y]); log(f"  {y} done")

KEEP=["race_id","馬名","bn","人気","odds","着","win","top2","top3","頭数",
      "p_win","p_top2","p_top3","q_win","q_top3","c_win","c_top2","c_top3","c_win_n",
      "EV_tan","kelly","ratio3","mr","pr","乖離","EV順","比率順"]
def prep(cur,prev,y):
    for tgt in ("win","top2","top3"):
        iso=IsotonicRegression(out_of_bounds="clip").fit(prev[f"p_{tgt}"],prev[tgt])
        cur[f"c_{tgt}"]=iso.predict(cur[f"p_{tgt}"])
    cur["c_win_n"]=cur["c_win"]/cur.groupby("race_id")["c_win"].transform("sum")
    cur["EV_tan"]=cur["c_win_n"]*cur["odds"]
    # ケリー比率: 人気薄ほど大きなエッジを要求する自然な指標
    cur["kelly"]=(cur["c_win_n"]*cur["odds"]-1)/np.clip(cur["odds"]-1,1e-9,None)
    cur["ratio3"]=cur["c_top3"]/cur["q_top3"]
    g=cur.groupby("race_id")
    cur["mr"]=g["p_top3"].rank(ascending=False,method="first")
    cur["pr"]=g["人気"].rank(method="first")
    cur["乖離"]=cur["pr"]-cur["mr"]
    cur["EV順"]=g["EV_tan"].rank(ascending=False,method="first")
    cur["比率順"]=g["ratio3"].rank(ascending=False,method="first")
    cur[KEEP].to_csv(f"bet_cache_{y}.csv",index=False)
    log(f"  bet_cache_{y}.csv 保存（{len(cur):,}行）")

log("較正して保存...")
for _y in (2021,2022,2023,2024,2025):
    prep(Y[_y],Y[_y-1],_y)
log("完了")
