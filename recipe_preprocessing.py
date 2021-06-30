#######################################
# レシピデータに対する前処理を行う関数を定義 #
#######################################

import MeCab
import re
import mojimoji
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

# 形態素解析を行う
def morphological_analysis(string):
    string = mojimoji.han_to_zen(string, ascii=False) # 半角カナを全角に変換
    mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    mecab.parse('') 
    
    parse = mecab.parse(string)
    lines = parse.split('\n')
    items = [re.split('[\t,]', line) for line in lines]
    
    del items[-2:] # 'EOS', 改行の削除
    return items

# 形態素を抽出
# 返り値: str
def morphome_extract(text):
    # 英語, 数字の削除
    text = mojimoji.zen_to_han(text, kana=False)
    text = re.sub('[a-zA-Z0-9]+', ' ', text)
    # 形態素解析
    items = morphological_analysis(text)
    new_text = ''
    for item in items:
        if item[1] != '記号':
            new_text += ' ' + item[0]
    new_text = new_text.strip()
    return new_text

# 文章をカタカナに変換
def text_to_katakana(text):
    if type(text) is str:
        items = morphological_analysis(text)
        string = ''
        for item in items:
            if len(item) >= 9:
                string += (' ' + item[8])
            else:
                string += (' ' + item[0])
        string = string.strip()
    else:
        string = np.nan
    return string

# カッコ内の文字列削除
# https://qiita.com/mynkit/items/d6714b659a9f595bcac8
def delete_brackets(s):
    """
    括弧と括弧内文字列を削除
    """
    """ brackets to zenkaku """
    table = {
        "(": "（",
        ")": "）",
        "<": "＜",
        ">": "＞",
        "{": "｛",
        "}": "｝",
        "[": "［",
        "]": "］",
        "~":"〜"
    }

    for key in table.keys():
        s = s.replace(key, table[key])
        
    #  全ての文字列がカッコで囲まれている時, カッコの中身を返す
    if (s[0] == '（' and s[-1] == '）') or (s[0] == '＜' and s[-1] == '＞') or (s[0] == '「' and s[-1] == '」') \
        or (s[0] == '【' and s[-1] == '】') or (s[0] == '｛' and s[-1] == '｝') or (s[0] == '〔' and s[-1] == '〕') \
        or (s[0] == '〈' and s[-1] == '〉') or (s[0] == '［' and s[-1] == '］') or (s[0] == '〜' and s[-1] == '〜'):
        
        return s[1:-1]
    
    """ delete zenkaku_brackets """
    l = ['（[^（|^）]*）', '【[^【|^】]*】', '＜[^＜|^＞]*＞', '［[^［|^］]*］',
         '「[^「|^」]*」', '｛[^｛|^｝]*｝', '〔[^〔|^〕]*〕', '〈[^〈|^〉]*〉', '〜[^〜|^〜]*〜']
    for l_ in l:
        s = re.sub(l_, "", s)
    """ recursive processing """
    return delete_brackets(s) if sum([1 if re.search(l_, s) else 0 for l_ in l]) > 0 else s

# カッコ内の文字列, 英語, 数字の削除
# symbol=Trueのとき, 記号を空白に変換
def remove_ascii_brackets(string, symbol=False):
    #print(string)
    # 全角の英語, 数字を半角に変換
    string = mojimoji.zen_to_han(string, kana=False)
    # 英語, 数字の削除
    string = re.sub('[a-zA-Z0-9]+', ' ', string)
    # 先頭, 末尾の空白削除
    string = string.strip()
    
    if len(string) != 0:
        # カッコの中身を削除
        string = delete_brackets(string)
        # 記号の削除
        if symbol:
            # 追加材料が複数の時の対策
            # 例) "ショートニングorバター"
            # いったんリストに変換して, 空白で分ける
            string_list = string.split(' ')
            temp_list = []
            for string in string_list:
                if len(string) != 0:
                    items = morphological_analysis(string)
                    new_string = ''
                    for item in items:
                        if item[1] == '記号':
                            new_string += ' '
                        else:
                            new_string += item[0]
                    new_string = new_string.strip()
                    temp_list.append(new_string)
            string = ' '.join(temp_list).strip()
    return string

# 名詞のみを抽出
# 返り値: 原型のstr, カタカナのstr
def noun_extract(text, replace=None, replace_hash=None):
    if type(text) is str:
        
        # 置換を行う
        if replace:
            text = replace_text(text, replace_hash=replace_hash)
            
        # 形態素解析
        items = morphological_analysis(text)
        
        prototype = '' # 原型
        katakana = '' # 読み方
        for item in items:
            if item[1] == '名詞':
                # 原型があれば原型
                if len(item) >= 8:
                    # 原型がない時その単語
                    if item[7] == '*':
                        prototype += (item[0] + ' ')
                    # 原型が英語の時その単語
                    elif item[7][0].isalnum():
                        prototype += (item[0] + ' ')
                    else:
                        prototype += (item[7] + ' ')
                # なければその単語
                else:
                    prototype += (item[0] + ' ')
                # 読みがあれば読み
                if len(item) >= 9:
                    katakana += (item[8] + ' ')
                # なければその単語
                else:
                    katakana += (item[0] + ' ')
        
        # 先頭と末尾の空白を削除
        prototype = prototype.strip()
        katakana = katakana.strip()
    else:
        return np.nan, np.nan
    
    return prototype, katakana

# 名詞に分類されない食材を置換
#  例) いか　→ イカ
def replace_text(string, replace_hash=None):
    if type(string) is str:    
        if not replace_hash:
            replace_hash = {
                'かぶ': 'カブ',
                'からし': '辛子',
                'たれ': 'タレ',
                'しお': '塩',
                'いか': 'イカ',
                '一味': 'イチミ',
                'あん': 'あんこ',
                'ふりかけ': 'フリカケ',
                'たら': '鱈',
                'なると': 'ナルト',
                'さわら': '鰆',
                'つくね': '捏',
                'ながいも': '長芋',
                'みず': '水',
                'ふ': 'フスマ',
                'ぶなぴー': 'シメジ',
            }
        if string in replace_hash:
            return replace_hash[string]
        else:
            if '回目' in string:
                return string.replace('回目', '')
            return string
    else:
        return np.nan

# 鮭の読み方をシャケにする
def sake_to_shake(row):
    if row['name_katakana'] == 'サケ':
        if '鮭' ==  row['name_prototype']:
            return 'シャケ'
        else:
            return row['name_katakana']
    else:
        return row['name_katakana']

# 動詞を抽出
# 返り値: str 空白区切り
def verb_extract(text, alpha_num=False):
    # 英語, 数字の削除
    if alpha_num:
        text = mojimoji.zen_to_han(text, kana=False)
        text = re.sub('[a-zA-Z0-9]+', ' ', text)
    
    # 形態素解析
    items = morphological_analysis(text)
    string_prototype = '' # 原型
    string_katakana = '' # カタカナ
    for item in items:
        if  item[1] == '動詞':
            # 原型があれば原型
            if len(item) >= 8:
                # 原型がない時その単語
                if item[7] == '*':
                    string_prototype += (' ' + item[0])
                else:
                    string_prototype += (' ' + item[7])
            # なければその単語
            else:
                string_prototype += (' ' + item[0])
    
    string_prototype = string_prototype.strip()
    string_katakana = text_to_katakana(string_prototype)
    
    return string_prototype, string_katakana

#　材料名に対して前処理を行う
def preprocessed_ingredients(df, test):
    # 欠損値の削除
    df = df.dropna(subset=['name']).reset_index(drop=True)
    # 記号, 数字, アルファベット, カッコ内の文字列を削除
    name_without_symbol = [remove_ascii_brackets(name, symbol=True) for name in (df['name'])]
    df['name_without_symbol'] = name_without_symbol 
    df = df[df['name_without_symbol'] != ''].reset_index(drop=True)
    
    # 名詞のみを抽出
    prototypes = [] # 名詞の原型
    katakanas = [] # 名詞の読み方(カタカナ)
    for name in tqdm(df['name_without_symbol']):
        prototype, katakana = noun_extract(name, replace=True)
        prototypes.append(prototype)
        katakanas.append(katakana)
        
    df['name_prototype'] = prototypes
    df['name_katakana'] = katakanas
    df  = df[df['name_prototype'] != ''].reset_index(drop=True)
    
    if test:
        # 鮭の読みをシャケに変換
        df['name_katakana'] = df.apply(sake_to_shake, axis=1)

    return df

# 食材の置換
def replace_name(name, test):
    # 置換単語の読み込み
    replace_word = pickle.load(open('data/replace_name.pickle', 'rb'))
    if name in replace_word[0]:
        return replace_word[0][name]
    
    if 'ヒキニク' in name and test:
        return 'ヒキニク'
    
    # 部分一致を置換
    for key, value in replace_word[1].items():
        if key in name:
            return value
    
    if 'コショウ' in name and 'ユズコショウ' not in name:
        return 'コショウ'

    if 'トリ' in name and 'トリガラ' not in name:
        return 'トリニク'

    if 'ゴマ' in name and name != 'ゴマアブラ' and name != 'ゴマダレ':
        return 'ゴマ'
   
    if 'ノウスギリ' in name:
        return name.replace('ノウスギリ', '')
    
    if 'ウスギリノ' in name:
        return name.replace('ウスギリノ', '')
    
    if 'ウスギリ' in name:
        return name.replace('ウスギリ', '')
    
    if 'オコノミノ' in name and name != 'オコノミヤキソース':
        return name.replace('オコノミノ', '')
    
    if 'オコノミデ' in name:
        return name.replace('オコノミデ', '')
    
    if 'ノミジンギリ' in name:
        return name.replace('ノミジンギリ', '')
    
    if 'ミジンギリ' in name:
        return name.replace('ミジンギリ', '')
    
    if 'レイトウノ' in name:
        return name.replace('レイトウノ', '')
    
    if 'レイトウ' in name:
        return name.replace('レイトウ', '')
    
    if 'リョウリノタメノ' in name:
        return name.replace('リョウリノタメノ', '')

    if 'リョウリヨウノ' in name:
        return name.replace('リョウリヨウノ', '')    

    if 'リョウリヨウ' in name:
        return name.replace('リョウリヨウ', '')
    
    if 'ノキリミ' in name:
        return name.replace('ノキリミ', '')
    
    if 'キリミ' in name:
        return name.replace('キリミ', '')
    
    if 'コクサンノ' in name and test:
        return name.replace('コクサンノ', '')
    
    if 'ノカンズメ' in name and test:
        return name.replace('ノカンズメ', '')
    
    if 'ノスリオロシ' in name and test:
        return name.replace('ノスリオロシ', '')
    
    if 'スリオロシタ' in name and test:
        return name.replace('スリオロシタ', '')
    
    if 'スリオロシノ' in name and test:
        return name.replace('スリオロシノ', '')
    
    if 'スリオロシ' in name and test:
        return name.replace('スリオロシ', '')

    if name[-2:] == 'ルイ':
        return name[:-2]
    
    # 1文字の単語を削除
    if len(name) == 1:
        return ''
    
    if name[-2:] == 'カン':
        return name[:-2]
    
    return name

# カタカナに変換
def name_katakana_replace(names, test):
    names = names.split(' ')
    new_names = ''
    for name in names:
        name = replace_name(name, test)
        if name != '':
            new_names += (' ' + name)
    new_names = new_names.strip()
    
    return new_names

# 食材名を１行の文字列に変換する
def ingredients_to_one_line(df, test):
    ingredients_list = []
    for index, row in tqdm(df.groupby('recipe_id')):
        ingredients_list.append([index, ' '.join(row['name_prototype'].values), ' '.join(row['name_katakana'])])
    df = pd.DataFrame(ingredients_list, columns=['recipe_id', 'name_prototype', 'name_katakana'])
    # カタカナの置換を行う
    df['ingredients_katakana'] = [name_katakana_replace(name, test) for name in df['name_katakana']]
    # 調味料リスト
    seasoning = ['ショウユ', 'シオ', 'サトウ', 'コショウ', 'サケ', 'ミズ', 'ゴマアブラ', 'ミリン', \
        'カタクリコ', 'サラダユ', 'オス', 'オユ', 'アブラ', 'オリーブオイル']
    # 調味料を削除
    df['ingredients_katakana_new'] = df['ingredients_katakana'].apply(lambda x: remove_ingredients(x, seasoning))
    df = df.dropna()
    df = df[df['ingredients_katakana_new'] != '']
    df = df.drop(columns=['ingredients_katakana'])
    df = df.rename(columns={'ingredients_katakana_new': 'ingredients_katakana'})
    
    return df

# 料理オントロジーの適用
def replace_otorogy(ingredients):
    # 料理オントロジーを取得
    ingredients_dict = pickle.load(open('data/ingredients_otorogy_dict.pickle', 'rb'))
    ingredients = ingredients.split(' ')
    new_ingredients = [ingredients_dict[ingredient] for ingredient in ingredients if ingredient in ingredients_dict]
    if len(new_ingredients) == 0:
        return np.nan
    else:
        return ' '.join(new_ingredients)

# 食材情報から特定の食材を削除
def remove_ingredients(ingredients, remove_list):
    ingredients = ingredients.split(' ')
    new_ingredients = [ingredient for ingredient in ingredients if ingredient not in remove_list]
    if len(new_ingredients) == 0:
        return np.nan
    else:
        return ' '.join(new_ingredients)

# 特定のカテゴリで多く使用される食材の削除
def remove_category_ingredients(df):
    remove_list = {
        0: ['ゴハン'],
        1: ['パスタ'],
        11: ['ナガネギ'],
        12: ['パン', 'コムギコ', 'タマゴ', 'タマネギ'],
        13: ['タマゴ'],
        14: ['キョウリキコ'],
        15: ['タマゴ']
    }

    new_ingredients = [remove_ingredients(ingredients, remove_list[category]) if category in remove_list \
        else ingredients for category, ingredients in zip(df['categories'], df['ingredients_katakana'])]
    
    df['ingredients_katakana'] = new_ingredients
    df = df.dropna()
    return df
        

# 追加食材の前処理
def preprocessed_add_ingredients(df):
    # 食材情報をカタカナに変換
    katakanas = [text_to_katakana(ingredient) for ingredient in tqdm(df['add_ingredients'])]
    # 食材情報を置換
    katakanas = [replace_name(katakana, False) for katakana in katakanas]
    df['add_ingredients_katakana'] = katakanas
    df = df.dropna(subset=['add_ingredients_katakana']).reset_index(drop=True)
    count_dict = {}
    for ingredient in df['add_ingredients_katakana']:
        if ingredient in count_dict:
            count_dict[ingredient] += 1
        else:
            count_dict[ingredient] = 1
    df['label'] = df['add_ingredients_katakana'].apply(lambda x: 1 if count_dict[x] > 2 else 0)
    df = df[df['label'] == 1]
    df = df.drop(columns=['label'])    
    # 調味料リスト
    seasoning = ['ショウユ', 'シオ', 'サトウ', 'コショウ', 'サケ', 'ミズ', 'ゴマアブラ', 'ミリン', \
        'カタクリコ', 'サラダユ', 'オス', 'オユ', 'アブラ', 'オリーブオイル']
    df['label'] = df['add_ingredients_katakana'].apply(lambda x: 1 if x in seasoning else 0)
    df = df[df['label'] == 0]
    df = df.drop(columns=['label'])
    return df

# 料理オントロジーの適用
def ingredients_to_otorogy(df, test=True):
    # 料理オントロジーを取得
    ingredients_dict = pickle.load(open('data/ingredients_otorogy_dict.pickle', 'rb'))
    
    df = df.dropna()
    
    if not test:
        df = remove_category_ingredients(df)
    
    # 食材情報にオントロジーを適用
    df['ingredients_katakana_new'] = df['ingredients_katakana'].apply(lambda x: replace_otorogy(x))
    df = df.dropna()
    df = df[df['ingredients_katakana_new'] != '']

    if test:
        df = df[df['add_ingredients_katakana'] != '']
        df['add_ingredients_katakana_new'] = df['add_ingredients_katakana'].apply(lambda x: \
            ingredients_dict[x] if x in ingredients_dict else np.nan)
        
        df = df.dropna()
        df = df[df['add_ingredients_katakana_new'] != '']
        df = df.reindex(columns=['recipe_id', 'title', 'ingredients_katakana_new', 'add_ingredients_katakana_new'])
        df = df.rename(columns={'ingredients_katakana_new': 'ingredients_katakana', \
            'add_ingredients_katakana_new': 'add_ingredients_katakana'})
    else:        
        df = df.dropna()
        df = df.reindex(columns=['recipe_id', 'title', 'ingredients_katakana_new', 'categories'])
        df = df.rename(columns={'ingredients_katakana_new': 'ingredients_katakana'})
        
    # 食材のカテゴリを取得
    otorogy_df = pd.read_csv('data/ingredients_otorogy.csv')
    # 調味料を取得
    seasoning_list = list(otorogy_df[otorogy_df['category'] == '調味料']['represent_katakana'].values)
    seasoning_list = sorted(set(seasoning_list), key=seasoning_list.index)

    # 調味料を削除
    df['ingredients_katakana_new'] = df['ingredients_katakana'].apply(lambda x: remove_ingredients(x, seasoning_list))
    df = df.dropna()
    df = df[df['ingredients_katakana_new'] != '']

    if test:
        df['labels'] = df['add_ingredients_katakana'].apply(lambda x: 1 if x in seasoning_list else 0)
        df[df['labels'] == 1]
        df = df[df['labels'] == 0].drop(columns=['labels'])

        df = df.reindex(columns=['recipe_id', 'title', 'ingredients_katakana_new', 'add_ingredients_katakana'])
        df = df.rename(columns={'ingredients_katakana_new': 'ingredients_katakana'})
    else:
        df = df.reindex(columns=['recipe_id', 'title', 'ingredients_katakana_new', 'categories'])
        df = df.rename(columns={'ingredients_katakana_new': 'ingredients_katakana'})
    
    df = df[df['ingredients_katakana'] != '']
    # 食材情報で同一の食材を削除
    df['ingredients_katakana'] = df['ingredients_katakana'].apply(lambda x: ' '.join(list(set(x.split(' ')))))
    # 食材数を取得
    df['ingredients_num'] = df['ingredients_katakana'].apply(lambda x: len(x.split(' ')))
    # 食材が2未満のレシピを削除
    df = df[df['ingredients_num'] > 1]
    df = df.drop(columns=['ingredients_num']).reset_index(drop=True)
    
    # 検索用データセットのとき重複を削除
    if not test:
        df = df[~df.duplicated()]

    return df
