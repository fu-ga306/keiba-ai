with open(r'c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai\model.py', 'r', encoding='utf-8') as f:
    c = f.read()

if 'LGB_RANK_PARAMS = {' in c:
    print('すでに定義済み')
else:
    rank = (
        '\nLGB_RANK_PARAMS = {\n'
        '    "objective": "lambdarank",\n'
        '    "metric": "ndcg",\n'
        '    "ndcg_eval_at": [3, 5],\n'
        '    "learning_rate": 0.03,\n'
        '    "num_leaves": 31,\n'
        '    "min_child_samples": 20,\n'
        '    "feature_fraction": 0.8,\n'
        '    "bagging_fraction": 0.8,\n'
        '    "bagging_freq": 1,\n'
        '    "verbose": -1,\n'
        '    "min_gain_to_split": 0.1,\n'
        '}\n'
    )
    c = c.replace('\ndef add_extra_features', rank + '\ndef add_extra_features', 1)
    with open(r'c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai\model.py', 'w', encoding='utf-8') as f:
        f.write(c)
    print('追加完了:', 'LGB_RANK_PARAMS = {' in c)
