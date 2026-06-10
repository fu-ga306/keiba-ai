with open(r'c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai\model.py', 'r', encoding='utf-8') as f:
    content = f.read()

# LambdaRankをmodelsリストに追加しない形に変更
old = '''        models.append(LambdaRankWrapper(rank_booster))
        print(f"  LambdaRank完了（{rank_booster.best_iteration}本）")
    except Exception as e:
        print(f"  LambdaRankスキップ: {e}")
        import traceback; traceback.print_exc()

    print(f"\\nアンサンブル: {len(models)}モデルの平均で予測")
    print(f"  ※LambdaRankを含む場合は着順ランキング学習も反映")

    # ── テストデータで評価 ──
    test_df = test_df.copy()
    # LambdaRankWrapperはpickle非対応のためnumpy配列で予測
    preds_list = []
    for m in models:
        try:
            preds_list.append(m.predict_proba(X_test)[:, 1])
        except Exception as e:
            print(f"  予測エラー（スキップ）: {e}")
    test_df["予測勝率スコア"] = np.mean(preds_list, axis=0) if preds_list else np.zeros(len(test_df))'''

new = '''        # LambdaRankはアンサンブルに含めず別途参考スコアとして保持
        lambda_wrapper = LambdaRankWrapper(rank_booster)
        print(f"  LambdaRank完了（{rank_booster.best_iteration}本）")
    except Exception as e:
        lambda_wrapper = None
        print(f"  LambdaRankスキップ: {e}")
        import traceback; traceback.print_exc()

    print(f"\\nアンサンブル: {len(models)}モデルの平均で予測（LGB+XGB+CatBoost）")
    if lambda_wrapper:
        print(f"  ※LambdaRankは参考スコアとして別途計算")

    # ── テストデータで評価 ──
    test_df = test_df.copy()
    test_df["予測勝率スコア"] = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)

    # LambdaRankの参考スコアを追加
    if lambda_wrapper:
        try:
            lambda_scores = lambda_wrapper.predict_proba(X_test)[:, 1]
            test_df["lambda_score"] = lambda_scores
        except Exception as e:
            print(f"  LambdaRankスコア計算エラー: {e}")'''

assert old in content, "対象なし"
content = content.replace(old, new, 1)

# pickle保存部分も修正
old2 = '''    # pickle保存（LambdaRankWrapperを除外してboosterとして保存）
    saveable_models = []
    lambda_booster  = None
    for m in models:
        if isinstance(m, LambdaRankWrapper):
            lambda_booster = m.booster
        else:
            saveable_models.append(m)

    save_dict = {"models": saveable_models, "use_cols": use_cols}
    if lambda_booster is not None:
        save_dict["lambda_booster"] = lambda_booster

    with open("model.pkl", "wb") as f:
        pickle.dump(save_dict, f)
    print(f"モデル保存完了 → model.pkl（通常モデル{len(saveable_models)}個" +
          (" + LambdaRankbooster）" if lambda_booster else "）"))'''

new2 = '''    # pickle保存（LGB+XGB+CatBoostの3モデル）
    save_dict = {"models": models, "use_cols": use_cols}
    if lambda_wrapper is not None:
        save_dict["lambda_booster"] = lambda_wrapper.booster

    with open("model.pkl", "wb") as f:
        pickle.dump(save_dict, f)
    print(f"モデル保存完了 → model.pkl（{len(models)}モデル" +
          (" + LambdaRank参考スコア）" if lambda_wrapper else "）"))'''

assert old2 in content, "pickle保存対象なし"
content = content.replace(old2, new2, 1)

with open(r'c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai\model.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("修正完了")
