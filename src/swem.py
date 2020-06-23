import gensim
import MeCab

mecab = MeCab.Tagger("-Owakati")


def get_word_list(document):
    document = document.lower()
    return list(filter("".__ne__, mecab.parse(document).split()))
