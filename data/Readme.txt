データセット作成に必要なデータ

ファイル内容

ingredients_otoroogy.csv:
料理オントロジーが保存されたcsvファイル
食材のカテゴリ, 食材名, 食材名のカタカナ, 同義語のカタカナ

ingredients_otorogy_dict.pikcle:
料理オントロジーの辞書
{同義語のカタカナ: 食材名のカタカナ}

replace_name.pickle:
食材名を置換する辞書
{元の食材名: 置換食材}

category_dict.pickle:
深さ2のカテゴリに属するカテゴリの辞書
{カテゴリi: カテゴリiに属するカテゴリのリスト}

category_recipes_id.pickle:
カテゴリの付与されているレシピIDのリスト