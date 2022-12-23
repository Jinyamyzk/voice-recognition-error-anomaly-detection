from transformers import BertForMaskedLM
from transformers import BertJapaneseTokenizer
import torch
from torch import nn

import pandas as pd
import copy
from tqdm import tqdm
import warnings

tqdm.pandas()
warnings.simplefilter("ignore")


def fill_mask(model,  tokenizer, masked_text):
    inputs = tokenizer(masked_text, return_tensors="pt").to(device)
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    with torch.no_grad():
        output = model(**inputs)
    predicted_token_ids = output.logits[0, mask_token_index]
    return predicted_token_ids

def preprocess_label(label):
    """ラベルの要素が文字列になっているので数字に直す"""
    table = str.maketrans({
    "[": "",
    "]": "",
    })
    label = [int(l.translate(table)) for l in label.split(",")]
    return torch.tensor(label)

def eval_model(row, model, tokenizer, softmax):
    """
    モデルを評価。AccuracyとRecallを返す
    """
    text = row["text"]
    label = row["label"]
    label = preprocess_label(label)

    MASK_TOKEN = tokenizer.mask_token
    tokens = tokenizer.tokenize(text)
    result = []
    for i in range(len(tokens)):
        tokens_ = copy.deepcopy(tokens)
        target = tokens_[i]
        token_id = tokenizer.convert_tokens_to_ids(target)
        tokens_[i] = MASK_TOKEN
        masked_text = "".join(tokens_).replace('#', '')
        preds = fill_mask(model, tokenizer, masked_text)
        probes = softmax(preds)
        top_k = torch.topk(probes[0], k=50)
        result.append(int(token_id in top_k.indices.tolist()))
    result = torch.tensor(result)
    len_label = len(label)
    accuracy = torch.sum(result==label) / len_label
    num_positive = torch.sum(label)
    label = torch.where(label < 0.5, -1, 1)
    recall = torch.sum(result==label) / num_positive
    return pd.Series([accuracy, recall])

def main(df):
    model = BertForMaskedLM.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    softmax = nn.Softmax(dim=1)

    model.to(device)
    model.eval()

    df[["accuracy", "recall"]] = df.progress_apply(eval_model, args=(model, tokenizer, softmax,), axis=1)
        
    print(f"Accuracy: {df.accuracy.mean()}, Recall: {df.recall.mean()}")


if __name__ == "__main__":
    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)
    print('-----start-------')

    df = pd.read_csv("livedoor_noised_data.tsv", sep="\t", names=["text", "label"])
    df = df.head(5) # For debugging
    main(df)