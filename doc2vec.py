#################################################
# doc2vecモデルの作成およびベクトルの取得する関数を定義 #
#################################################

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from recipe_preprocessing import morphome_extract
from connect_sql import search_by_id

# doc2vecモデルの作成
def get_doc2vec(save=False):
    if save:
        # 学習用のレシピの読み込み
        recipe_id_list = pickle.load(open('data/recipe_id_list.pickle', 'rb'))
        steps_df = search_by_id('steps', recipe_id_list)
        steps_df = steps_df.dropna().reset_index(drop=True)
        # 調理手順の前処理
        count_list = []
        for index, row in tqdm(steps_df.groupby('recipe_id')):
            count_list.append([index, row.shape[0]])
        steps_count_df = pd.DataFrame(count_list, columns=['recipe_id', '#_of_steps'])
        del count_list
        # 調理手順の工程が2以上10未満のレシピを使用
        steps_count_df = steps_count_df[(steps_count_df['#_of_steps'] > 1) & (steps_count_df['#_of_steps'] < 10)]
        steps_list = []
        for index, row in tqdm(steps_df.groupby('recipe_id')):
            steps = ''
            for i, memo in enumerate(row['memo']):
                steps += memo
            steps_list.append([index, steps])
        
        steps_df = pd.DataFrame(steps_list, columns=['recipe_id', 'steps'])
        del steps_list

        steps_df = pd.merge(steps_count_df, steps_df, on='recipe_id', how='left')
        steps_df = steps_df.drop(columns=['#_of_steps']).reset_index(drop=True)

        steps_list = [morphome_extract(steps) for steps in tqdm(steps_df['steps'])]
        steps_df['steps_morphome'] = steps_list
        steps_df = steps_df[steps_df['steps_morphome'] != '']
        steps_df = steps_df.reset_index(drop=True)

        sentences = []
        # 学習できる形に変換
        for text in steps_df['steps_morphome']:
            text_list = text.split(' ')
            sentences.append(text_list)

        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]
        model = Doc2Vec(documents, vector_size=300, window=5)

        pickle.dump(model, open('data/doc2vec_model.pickle', 'wb'))
    else:
        return pickle.load(open('data/doc2vec_model.pickle', 'rb'))

# 料理手順から料理レシピのベクトルを出力
def get_vector(steps, morphome=True, model=None):
    # 形態素解析を行う
    if morphome:
        steps = morphome_extract(steps)
    steps = steps.split(' ')
    # pv-dmモデルの読み込み
    if model is None:
        model = get_doc2vec()

    return model.infer_vector(steps)

# コサイン類似度を計算
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# コサイン類似度を計算して, ソートを行う
# vector: 対象ベクトル
# vectors_hash: {recipe_id: vector}
def sorted_cos_sim(v1, vectors_hash):
    cos_sim_hash = {}
    for recipe_id, v2 in vectors_hash.items():
        # コサイン類似度を計算
        cos_sim_hash[recipe_id] = cos_sim(v1, v2)
        
    # コサイン類似度が高い順にソート
    cos_sim_hash = sorted(cos_sim_hash.items(), reverse=True, key=lambda x:x[1])
    return cos_sim_hash

# 対象レシピおよび学習用レシピのベクトルを取得
def get_vectors(save=False, doc2vec=False):
    if save:
        if doc2vec:
            get_doc2vec(save=True)
            model = get_doc2vec()
        else:
            model = get_doc2vec()
        
        # ベクトル保存用のフォルダを作成
        directory = 'data/vectors'
        if not os.path.exists(directory):
            os.mkdir(directory)
        
        # テスト用のレシピのベクトルを取得
        test_data_df = pd.read_csv('data/test_data_recipes.csv')    
        test_recipe_id = list(test_data_df['recipe_id'].values)
        test_recipe_id = sorted(set(test_recipe_id), key=test_recipe_id.index)
        test_steps = search_by_id('steps', test_recipe_id)
        test_steps = test_steps.dropna(subset=['memo']).reset_index(drop=True)

        steps_list = []
        for index, row in tqdm(test_steps.groupby('recipe_id')):
            steps = ''
            for memo in row['memo']:
                steps += memo
            steps_list.append([index, steps])
        test_steps = pd.DataFrame(steps_list, columns=['recipe_id', 'steps'])

        test_vectors = {recipe_id: get_vector(steps, model=model) for recipe_id, steps in tqdm(zip(test_steps['recipe_id'], \
            test_steps['steps']), total=test_steps.shape[0])}
        
        pickle.dump(test_vectors, open('data/vectors/test_data_vectors.pickle', 'wb'))
        
        # 学習用のレシピのベクトルを取得
        category_df = pd.read_csv('data/category_recipes.csv')
        category_recipe_id = list(category_df['recipe_id'].values)
        category_steps = search_by_id('steps', category_recipe_id)
        category_steps = category_steps.dropna(subset=['memo']).reset_index(drop=True)

        steps_list = []
        for index, row in tqdm(category_steps.groupby('recipe_id')):
            steps = ''
            for memo in row['memo']:
                steps += memo
            steps_list.append([index, steps])
        category_steps = pd.DataFrame(steps_list, columns=['recipe_id', 'steps'])

        category_vectors = {recipe_id: get_vector(steps, model=model) for recipe_id, steps in tqdm(zip(category_steps['recipe_id'], \
            category_steps['steps']), total=category_steps.shape[0])}
        
        category_vectors_dict = {i: {} for i in range(16)}
        for recipe_id, category in zip(category_df['recipe_id'], category_df['categories']):
            category_vectors_dict[category][recipe_id] = category_vectors[recipe_id]
        
        for category, vectors in category_vectors_dict.items():
            pickle.dump(vectors, open('data/vectors/category_recipes_vectors_' + str(category) + '.pickle', 'wb'))
        
    else:
        test_data_vectors = pickle.load(open('data/vectors/test_data_vectors.pickle', 'rb'))
        category_vectors = [pickle.load(open('data/vectors/category_recipes_vectors_' + str(i) + '.pickle', 'rb')) \
            for i in range(16)]
        return test_data_vectors, category_vectors
