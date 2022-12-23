import re
from pykakasi import kakasi
import random
import urllib
import json
import glob
import pandas as pd
from tqdm import tqdm

from transformers import BertJapaneseTokenizer

from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.database.dict import DictDatabase
from simstring.searcher import Searcher

failure_count = 0

tqdm.pandas()
mykakasi = kakasi()
tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

def init_simstring_db(tokenizer):
    """
    Adding the vocablary of BERT tokenizer to simstring database
    """
    db = DictDatabase(CharacterNgramFeatureExtractor(1))
    ids = range(tokenizer.vocab_size)
    ids = ids[5:]
    for id in ids:
        word = tokenizer.convert_ids_to_tokens([id])[0]
        word = word.replace("#", "")
        word_yomi = kanji_to_hiragana(word)
        db.add(word_yomi)
    return db

def remove_symbol(text):
    text = re.sub(",", "", text)
    text = re.sub("\[","(",text)
    text = re.sub("\]",")",text)
    pattern = "\(.*?\)|\<.*?\>|《.*?》|\{.*?\}|【|】|#|'|…|“|”|=|‘|’|>|、|。|\?|「|」|『|』"
    text = re.sub(pattern, "", text)
    return text

def kanji_to_hiragana(word):
    result = mykakasi.convert(word)
    return result[0]["hira"]

def is_kanji(word):
    not_kanji = re.compile(r'[ぁ-んーァ-ンヴーｧ-ﾝa-zA-Z0-9０-９Ａ-Ｚ]+')
    return not not_kanji.search(word)

def hiragana_to_kanji(word_yomi):
    url = "http://www.google.com/transliterate?"
    param = {'langpair':'ja-Hira|ja','text':word_yomi}
    paramStr = urllib.parse.urlencode(param)
    readObj = urllib.request.urlopen(url + paramStr)
    response = readObj.read()
    data = json.loads(response)
    fixed_data = json.loads(json.dumps(data[0], ensure_ascii=False))[1]
    shuffled = random.sample(fixed_data, len(fixed_data))
    for word in shuffled:
        if is_kanji(word):
            result = word
            return word
    return word_yomi

def simstring_noise(word, searcher):
    word_yomi = kanji_to_hiragana(word)
    result = searcher.search(word_yomi, 0.8)
    noised = random.choice(result)
    # noised_kanji = hiragana_to_kanji(noised)
    # return noised_kanji
    return noised


def noise(text, searcher):
    """
    noise text.
    return noised text and label.
    """
    label = []
    noised_text = []
    text_split = tokenizer.tokenize(text)
    text_split = [word for word in text_split if word != "[UNK]"]
    for word in text_split:
        if len(word) >= 2 and random.uniform(0,1) <= 0.2:
            noised = simstring_noise(word.replace("#", ""), searcher)
            noised_len = len(tokenizer.tokenize(noised))
            noised_text.append(noised)
            label += [1] * noised_len
        else:
            noised_text.append(word)
            label += [0]
    noised_text = "".join(noised_text).replace("#", "")
    n_t_s = tokenizer.tokenize(noised_text)
    noised_text_len = len(n_t_s)
    if noised_text_len != len(label):
        text = "".join(text_split).replace("#", "")
        label = [0] * len(text_split)
        return pd.Series([text, label])
    else:
        return pd.Series([noised_text, label])

def convert_series_to_list(label):
    table = str.maketrans({
        "[": "",
        "]": "",
        "'": "",
        })
    label = list(map(int,str(label).translate(table).split(",")))
    return label

def concat_5_sentences(df):
    new_df = []
    for i in range(len(df)//5):
        idx = i*5
        sentences_concat = "".join(df.loc[idx:idx+4, "noised_sentence"].tolist())
        labels = [convert_series_to_list(label) for label in df.loc[idx:idx+4, "label"].tolist()]
        labels_concat = []
        for label in labels:
            labels_concat += label
        sentences_len = len(tokenizer.tokenize(sentences_concat))
        if sentences_len != len(labels_concat): # sentenceの長さとラベルの長さが違えば失敗
            global failure_count
            failure_count += 1
            continue
        if sentences_len > 510:
            sentences_concat = tokenizer.tokenize(sentences_concat)[:510]
            labels_concat = labels_concat[:510]
        new_df.append([sentences_concat, labels_concat])
    new_df = pd.DataFrame(new_df, index=None, columns=["sentence", "label"])
    return new_df

def main():
    db = init_simstring_db(tokenizer)
    searcher = Searcher(db, CosineMeasure())

    files = glob.glob(f"livedoor/*.tsv")
    # files = ["livedoor/dokujo-tsushin-6289369.tsv"]
    for file in tqdm(files, desc="[Loading files]"):
        # print(file.split("/")[-1])
        df = pd.read_csv(file, sep="\t", names=["sentence"], index_col=None)
        df = df[df["sentence"]!=""]

        df[["noised_sentence", "label"]] = df["sentence"].apply(noise, args=(searcher,))
        df = df[df["noised_sentence"]!=""]
        df.reset_index(inplace=True, drop=True)

        new_df = concat_5_sentences(df)
        new_df.to_csv("livedoor_noised_data.tsv", sep="\t", index=False, header=False, mode="a")
    
    print(f"Failure Count: {failure_count}")

if __name__ == "__main__":
    main()
