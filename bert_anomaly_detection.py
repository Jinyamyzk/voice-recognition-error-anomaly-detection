from transformers import BertForMaskedLM
from transformers import BertJapaneseTokenizer
import torch
from torch import nn

import pandas as pd
import copy
import csv
import argparse
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

def eval_model(row, model, tokenizer, softmax, top_k):
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
        top_k_words = torch.topk(probes[0], k=top_k)
        result.append(int(token_id not in top_k_words.indices.tolist()))
    result = torch.tensor(result)
    len_label = len(label)
    accuracy = torch.sum(result==label) / len_label
    num_positive = torch.sum(label)
    label = torch.where(label < 0.5, -1, 1)
    recall = torch.sum(result==label) / num_positive if num_positive != 0 else 0.0 # Avoid ZeroDivisionError
    precision = torch.sum(result==label) / torch.sum(result) if torch.sum(result) != 0 else 0.0
    return pd.Series([accuracy, precision, recall])

def main(df, model_path):
    model = BertForMaskedLM.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    if model_path:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    softmax = nn.Softmax(dim=1)

    model.to(device)
    model.eval()

    results = []
    topk_candidates = [10, 20, 30, 40, 50, 100]
    for topk in topk_candidates:
        df[["accuracy", "precision", "recall"]] = df.progress_apply(eval_model, args=(model, tokenizer, softmax, topk,), axis=1)
        accuracy = df.accuracy.mean()
        precision = df.precision.mean()
        recall = df.recall.mean()
        f1 = 2 * precision * recall / (precision + recall)
        f2 = 5 * precision * recall / (4 * precision + recall)
        results.append([topk, accuracy, precision, recall, f1, f2])
    for r in results:
        print(f"TopK: {r[0]}, Accuracy: {r[1]:.4f}, Precision: {r[2]:.4f},Recall: {r[3]:.4f}, F1: {r[4]:.4f}, F2: {r[5]:.4f}")
    with open("result.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["TopK", "Accuracy", "Precision", "Recall", "F1", "F2"])
        writer.writerows(results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path")
    args = parser.parse_args()

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)
    print('-----start-------')
    df = pd.read_csv("livedoor/test.tsv", sep="\t", names=["text", "label"])
    df = df.head(100) # For debugging
    main(df, args.model_path)