from transformers import BertForMaskedLM
from transformers import BertJapaneseTokenizer
import torch
from torch import nn

import pandas as pd
import copy
import warnings
warnings.simplefilter("ignore")

def fill_mask(model,  tokenizer, masked_text):
    inputs = tokenizer(masked_text, return_tensors="pt").to(device)
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    with torch.no_grad():
        output = model(**inputs)
    predicted_token_ids = output.logits[0, mask_token_index]
    return predicted_token_ids

def eval_model(text, label, model, tokenizer, softmax):
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
    label = torch.tensor(label)
    accuracy = torch.sum(result==label) / len_label
    num_positive = torch.sum(label)
    label = torch.where(label < 0.5, -1, 1)
    recall = torch.sum(result==label) / num_positive

    # return pd.Series([accuracy, recall])
    return accuracy, recall



def main():
    model = BertForMaskedLM.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    softmax = nn.Softmax(dim=1)

    model.to(device)
    model.eval()

    text = "どうして男の人って、テレビをつけたまま寝るんだろう契約社員の佳子さんが、とある飲み会でぽろっとつぶやいたところ、その場にいたほぼ全員の女性が、そうそうホント、やくめてほしいよねからといっせいやくにす同調したという周囲の女性たちの反応に、TVをつけっぱなしで寝る男にイラッとしているのは自分だけではない、と意を強くした佳子さん私の彼は、すごくテレビ好きなんですよ"
    label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    accuracy, recall =  eval_model(text, label, model, tokenizer, softmax)

    
    print(f"Accuracy: {accuracy}, Recall: {recall}")

if __name__ == "__main__":
    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)
    print('-----start-------')
    main()