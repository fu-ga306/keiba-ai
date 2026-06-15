with open(r'c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai\model.py', 'r', encoding='utf-8') as f:
    content = f.read()

RANK_PARAMS = '''
LGB_RANK_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [3, 5],
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "min_gain_to_split": 0.1,
}
'''

old = '\ndef add_extra_features(df):'
new = RANK_PARAMS + '\ndef add_extra_features(df):'

if 'LGB_RANK_PARAMS' in content:
    print('すでに定義済みです')
elif old in content:
    content = content.replace(old, new, 1)
    with open(r'c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai\model.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('追加完了')
else:
    print('対象なし')