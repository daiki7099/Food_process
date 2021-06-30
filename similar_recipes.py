################################
# 類似レシピ取得するための関数を定義 #
################################

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from doc2vec import get_vectors, sorted_cos_sim

# Jaccard距離を取得
def jaccard_distance(list1, list2):
    # リストを集合に変換
    set1 = set(list1)
    set2 = set(list2)

    # 和集合
    set_union = set1.union(set2)
    # 積集合
    set_intersection = set1.intersection(set2)

    return len(set_intersection) / len(set_union)

# カテゴリーごとの類似レシピを取得
def get_similar_recipes_category(save=False, vector=False, doc2vec=False):
    if save:
        # 類似レシピ用のフォルダを作成
        directory = 'data/similar_recipes'
        if not os.path.exists(directory):
            os.mkdir(directory)
        
        # 対象レシピの読み込み
        test_data_df = pd.read_csv('data/test_data_recipes.csv')
        # 対象レシピの食材情報を取得
        test_recipes_ingredients = {recipe_id: ingredients.split(' ') for recipe_id, ingredients \
            in zip(test_data_df['recipe_id'], test_data_df['ingredients_katakana'])}
        # ベクトルの取得(pickleで保存)
        if vector:
            get_vectors(save=True, doc2vec=doc2vec)
        # 対象レシピのベクトルの読み込み
        test_data_vectors = pickle.load(open('data/vectors/test_data_vectors.pickle', 'rb'))
        test_recipe_id_list = list(test_data_vectors.keys())
        # カテゴリー付きのレシピの読み込み
        category_df = pd.read_csv('data/category_recipes.csv')
        # カテゴリー付きのレシピの食材情報を取得
        category_recipes_ingredients = {recipe_id: ingredients.split(' ') for recipe_id, ingredients \
            in zip(category_df['recipe_id'], category_df['ingredients_katakana'])}
        # カテゴリー付きのレシピのベクトルの読み込み
        category_vectors = [pickle.load(open('data/vectors/category_recipes_vectors_' + str(i) + '.pickle', 'rb')) \
                        for i in range(16)]
        # 不要なメモリーの開放
        del test_data_df
        del category_df

        #　類似レシピの情報を保存
        similar_recipes_hash = {}
        count = 0
        for test_recipe_id in tqdm(test_recipe_id_list):
            test_ingredients = test_recipes_ingredients[test_recipe_id]
            
            # カテゴリーごとのコサイン類似度を取得
            cos_sim_hash = {i: sorted_cos_sim(test_data_vectors[test_recipe_id], category_vectors[i])[:300] for i in range(16)}
            
            # カテゴリーごとに保存用の変数
            category_hash = {}
            # Jaccard距離を取得
            for category, category_recipes in cos_sim_hash.items():
                # 類似レシピの保存用の変数
                # [レシピID, コサイン類似度, Jaccard距離]
                similar_recipes = []
                for recipe_id, cos_sim in category_recipes:
                    # カテゴリー付きレシピの食材情報を取得
                    category_ingredients = category_recipes_ingredients[recipe_id]
                    jaccard = jaccard_distance(test_ingredients, category_ingredients)
                    similar_recipes.append([recipe_id, cos_sim, jaccard])
                # カテゴリーごとに保存
                category_hash[category] = similar_recipes
            #　類似レシピを保存
            similar_recipes_hash[test_recipe_id] = category_hash
        
        pickle.dump(similar_recipes_hash, open('data/similar_recipes/category_similar_recipes.pickle', 'wb'))

        save_similar_recipes()

    else:
        return pickle.load(open('data/similar_recipes/category_similar_recipes.pickle', 'rb'))

# カテゴリーを考慮しない類似レシピの抽出
def get_similar_recipes_not_category(save=False):
    if save:
        # 対象レシピの読み込み
        test_data_df = pd.read_csv('data/test_data_recipes.csv')
        # 対象レシピの食材情報を取得
        test_recipes_ingredients = {recipe_id: ingredients.split(' ') for recipe_id, ingredients \
            in zip(test_data_df['recipe_id'], test_data_df['ingredients_katakana'])}
        # 対象レシピのベクトルの読み込み
        test_data_vectors = pickle.load(open('data/vectors/test_data_vectors.pickle', 'rb'))
        test_recipe_id_list  = list(test_data_vectors.keys())
        # カテゴリー付きのレシピの読み込み
        category_df = pd.read_csv('data/category_recipes.csv')
        # カテゴリー付きのレシピの食材情報を取得
        category_recipes_ingredients = {recipe_id: ingredients.split(' ') for recipe_id, ingredients \
            in zip(category_df['recipe_id'], category_df['ingredients_katakana'])}
        # カテゴリー付きのレシピのベクトルの読み込み
        category_vectors = {}
        for i in range(16):
            vectors = pickle.load(open('data/vectors/category_recipes_vectors_' + str(i) + '.pickle', 'rb'))
            category_vectors.update(vectors)
        
        # 不要なメモリーの開放
        del test_data_df
        del category_df
        
        similar_recipes_hash = {}
        count = 0
        for test_recipe_id in tqdm(test_recipe_id_list):
            # 対象レシピの食材情報を取得
            test_ingredients = test_recipes_ingredients[test_recipe_id]

            # コサイン類似度の高いレシピを取得
            cos_sim_hash = sorted_cos_sim(test_data_vectors[test_recipe_id], category_vectors)[:300]
            # 類似レシピの保存用の変数
            # [レシピID, コサイン類似度, Jaccard距離]
            similar_recipes = []
            # Jaccard距離を取得
            for recipe_id, cos_sim in cos_sim_hash:
                category_ingredients = category_recipes_ingredients[recipe_id]
                jaccard = jaccard_distance(test_ingredients, category_ingredients)
                similar_recipes.append([recipe_id, cos_sim, jaccard])
            # 類似レシピを保存
            similar_recipes_hash[test_recipe_id] = similar_recipes
        
        pickle.dump(similar_recipes_hash, open('data/similar_recipes/not_category_similar_recipes.pickle', 'wb'))
    else:
        return pickle.load(open('data/similar_recipes/not_category_similar_recipes.pickle', 'rb'))

# カテゴリ除外を行なった類似レシピを保存
def save_similar_recipes(category=True):
    # フォルダの作成
    directory = 'data/similar_recipes/similar_recipes_jaccard'
    if not os.path.exists(directory):
        os.mkdir(directory)

    df = pd.read_csv('data/test_data_recipes.csv')
    df = df[~df.duplicated(subset='recipe_id')]
    n_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    if category:
        similar_recipes_dict = get_similar_recipes_category()
        
        for n in tqdm(n_list):
            new_similar_recipes = {}
            for recipe_id, ingredients in tqdm(zip(df['recipe_id'], df['ingredients_katakana']), \
                total=df.shape[0]):
                
                ingredients = ingredients.split(' ')
                jaccard_upper = len(ingredients) / (len(ingredients) + 1)

                if recipe_id in similar_recipes_dict:
                    similar_recipes = similar_recipes_dict[recipe_id]
                else:
                    print('error!')
                    print('recipe_id "{}" is not found!'.format(recipe_id))
                    break
                
                category_similar_recipes = {}
                for category, similar_recipe in similar_recipes.items():
                    # 類似レシピの抽出 (jaccard距離が閾値より大きい)
                    new_similar_recipe = [[_recipe_id, cos_sim, jaccard] for _recipe_id, cos_sim, jaccard in similar_recipe \
                        if jaccard > 0 and jaccard < jaccard_upper]
                    if len(new_similar_recipe) > n:
                        category_similar_recipes[category] = new_similar_recipe
                
                new_similar_recipes[recipe_id] = category_similar_recipes
            pickle.dump(new_similar_recipes, open('data/similar_recipes/similar_recipes_jaccard/similar_recipes_' \
                + str(n) + '.pickle', 'wb'))
        
    else:
        similar_recipes_dict = get_similar_recipes_not_category()
        new_similar_recipes = {}
        for recipe_id, ingredients in tqdm(zip(df['recipe_id'], df['ingredients_katakana']), \
            total=df.shape[0]):
            
            ingredients = ingredients.split(' ')
            jaccard_upper = len(ingredients) / (len(ingredients) + 1)

            if recipe_id in similar_recipes_dict:
                similar_recipes = similar_recipes_dict[recipe_id]
            else:
                print('error!')
                print('recipe_id "{}" is not found!'.format(recipe_id))
            
            new_similar_recipe = [[_recipe_id, cos_sim, jaccard] for _recipe_id, cos_sim, jaccard in similar_recipes \
                if jaccard > 0 and jaccard < jaccard_upper]            
            
            new_similar_recipes[recipe_id] = new_similar_recipe

        pickle.dump(new_similar_recipes, \
            open('data/similar_recipes/similar_recipes_jaccard/not_category_similar_recipes.pickle', 'wb'))
