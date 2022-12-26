import torch
from torchtext.legacy import data
import torch.optim as optim
from torch import nn
from transformers import BertForMaskedLM
from transformers import BertJapaneseTokenizer
import numpy as np

tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
MAX_LENGTH = 512

def tokenizer_512(input_text):
    """torchtextのtokenizerとして扱えるように、512単語のpytorchでのencodeを定義。ここで[0]を指定し忘れないように"""
    return tokenizer.encode(input_text, max_length=MAX_LENGTH, truncation=True, return_tensors='pt')[0]

def mask_text(text_ids):
    input_ids = []
    for inp in text_ids.numpy():
        actual_tokens = list(set(range(512)) - set(np.where((inp==1) | (inp==2) | (inp==3)|(inp==0))[0].tolist()))
        num_of_token_to_mask = int(len(actual_tokens)*0.15)
        token_to_mask = np.random.choice(np.array(actual_tokens), size=num_of_token_to_mask, replace=False).tolist()
        inp[token_to_mask] = 4
        input_ids.append(inp)
    input_ids = torch.tensor(input_ids)
    return input_ids



def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)
    print('-----start-------')

    # ネットワークをGPUへ
    net.to(device)
    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # ミニバッチのサイズ
    batch_size = dataloaders_dict["train"].batch_size

     # epochのループ
    for epoch in range(num_epochs):
        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに
        
            epoch_loss = 0.0  # epochの損失和
            iteration = 1
            # データローダーからミニバッチを取り出すループ
            for batch in (dataloaders_dict[phase]):
                labels = batch.Text[0].unsqueeze(1).to(device)
                inputs = mask_text(batch.Text[0]).to(device)
                attn_mask = torch.where(inputs==0, 0, 1).to(device) # attention maskの作成
            
            # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(input_ids=inputs, attention_mask=attn_mask).logits
                    print(outputs.size())

                    loss = criterion(outputs, labels)
                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if (iteration % 10 == 0):  # 10iterに1度、lossを表示
                            print('イテレーション {} || Loss: {:.4f} || 10iter.'.format(
                                iteration, loss.item()))
                        

                    iteration += 1
                    # 損失と正解数の合計を更新
                    epoch_loss += loss.item() * batch_size
            # epochごとのlossと正解率
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            print('Epoch {}/{} | {:^5} |  Loss: {:.4f}'.format(epoch+1, num_epochs, phase, epoch_loss))
    return net

def main():
    TEXT = data.Field(sequential=True, tokenize=tokenizer_512, use_vocab=False, lower=False,
                                include_lengths=True, batch_first=True, fix_length=MAX_LENGTH, pad_token=0)

    dataset_train, dataset_valid, dataset_test = data.TabularDataset.splits(
        path="livedoor4finetune", train="train.tsv", validation="valid.tsv",test="test.tsv", format="tsv", fields=[
            ("Text", TEXT)])
    
    # DataLoaderを作成します（torchtextの文脈では単純にiteraterと呼ばれています）
    batch_size = 8 # BERTでは16、32あたりを使用する

    dl_train = data.Iterator(
        dataset_train, batch_size=batch_size, train=True)

    dl_valid = data.Iterator(
        dataset_valid, batch_size=batch_size, train=False, sort=False)

    dl_test = data.Iterator(
        dataset_test, batch_size=batch_size, train=False, sort=False)

    # 辞書オブジェクトにまとめる
    dataloaders_dict = {"train": dl_train, "val": dl_valid}

    net = BertForMaskedLM.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    # 訓練モードに設定
    net.train()

    optimizer = optim.Adam([
        {"params": net.parameters(), "lr": 5e-5},
    ])

    criterion = nn.CrossEntropyLoss()

    num_epochs = 3
    net_trained = train_model(net, dataloaders_dict, criterion, optimizer, num_epochs)

    # モデルの保存
    torch.save(net_trained.state_dict(), "model/model_trained.pt")






if __name__ == "__main__":
    main()