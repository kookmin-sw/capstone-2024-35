import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    TrainerCallback
)
from sklearn.metrics import accuracy_score

# GPU 사용 가능 확인 및 device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# SentimentDataset 클래스 정의
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# CSV 파일 로드
df = pd.read_csv('/content/drive/MyDrive/dataset_hr.csv')  # 'your_dataset.csv'를 실제 파일 경로로 변경하세요.

# 데이터 전처리
texts = df['person_corpus'].tolist()
labels = df['label'].tolist()  # 라벨 데이터는 정수로 변환되어 있어야 합니다.

# 토크나이저 로딩
tokenizer_roberta = AutoTokenizer.from_pretrained("klue/roberta-large")
tokenizer_koelectra = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

# 데이터셋 토큰화
encodings_roberta = tokenizer_roberta(texts, truncation=True, padding=True, max_length=128)
encodings_koelectra = tokenizer_koelectra(texts, truncation=True, padding=True, max_length=128)

# 데이터셋 생성
dataset_roberta = SentimentDataset(encodings_roberta, labels)
dataset_koelectra = SentimentDataset(encodings_koelectra, labels)

# 로드할 RoBERTa 모델 경로
roberta_model_path = "/content/drive/MyDrive/robust_model/roberta_model"
# 로드할 Electra 모델 경로
electra_model_path = "/content/drive/MyDrive/robust_model/electra_model"

# 저장된 모델 로드
model_roberta = AutoModelForSequenceClassification.from_pretrained(roberta_model_path)
model_koelectra = AutoModelForSequenceClassification.from_pretrained(electra_model_path)


# CustomTrainer 클래스 정의
class CustomTrainer(Trainer):
    def __init__(self, *args, label_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_weights = label_weights if label_weights is not None else {}

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.get('logits')
        labels = inputs.get('labels')
        if self.label_weights:
            weight = torch.tensor([self.label_weights.get(label.item(), 1.0) for label in labels]).to(labels.device)
            loss_fct = CrossEntropyLoss(weight=weight)
        else:
            loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=100,
    per_device_train_batch_size=16,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# 라벨별 가중치 설정
label_weights = {0: 8.0, 1: 2.5, 2: 15.0, 3: 1.0}

# CustomTrainer 인스턴스 생성
trainer_roberta = CustomTrainer(
    model=model_roberta,
    args=training_args,
    train_dataset=None,  # dataset_roberta는 사용자가 정의해야 함
    label_weights=label_weights,
)

trainer_electra = CustomTrainer(
    model=model_koelectra,
    args=training_args,
    train_dataset=None,  # dataset_electra는 사용자가 정의해야 함
    label_weights=label_weights,
)


# logits를 확률로 변환하는 함수
def logits_to_probs(logits):
    return torch.softmax(logits, dim=-1)


class EnsembleModel(nn.Module):
    def __init__(self, model_roberta, model_koelectra):
        super(EnsembleModel, self).__init__()
        self.roberta_model = model_roberta
        self.koelectra_model = model_koelectra
        self.star1 = nn.Parameter(torch.tensor(0.5))
        self.star2 = nn.Parameter(torch.tensor(0.5))

    def forward(self, texts, tokenizer_roberta, tokenizer_koelectra, device):
        self.star1.data = torch.clamp(self.star1.data, 0.0, 1.0)
        self.star2.data = torch.clamp(self.star2.data, 0.0, 1.0)

        # Tokenize inputs
        encodings_roberta = tokenizer_roberta(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
        encodings_koelectra = tokenizer_koelectra(texts, truncation=True, padding=True, max_length=128,
                                                  return_tensors='pt')

        roberta_input_ids = encodings_roberta['input_ids'].to(device)
        roberta_attention_mask = encodings_roberta['attention_mask'].to(device)

        koelectra_input_ids = encodings_koelectra['input_ids'].to(device)
        koelectra_attention_mask = encodings_koelectra['attention_mask'].to(device)

        # Get model outputs
        roberta_outputs = self.roberta_model(roberta_input_ids, attention_mask=roberta_attention_mask)
        koelectra_outputs = self.koelectra_model(koelectra_input_ids, attention_mask=koelectra_attention_mask)

        # Convert logits to probabilities
        roberta_probs = F.softmax(roberta_outputs.logits, dim=1)
        koelectra_probs = F.softmax(koelectra_outputs.logits, dim=1)

        # Ensemble probabilities
        ensemble_probs = (roberta_probs + koelectra_probs) / 2

        final_labels = []

        for probs in ensemble_probs:
            pred_label = torch.argmax(probs).item()
            if pred_label == 1 and probs[pred_label] > self.star1:
                pred_label = 3
            elif pred_label == 0 and probs[pred_label] > self.star2:
                pred_label = 2
            final_labels.append(pred_label)

        return torch.tensor(final_labels).to(device), (roberta_outputs.logits + koelectra_outputs.logits) / 2


from torch.optim import Adam

# 데이터셋 및 훈련 설정
ensemble_model = EnsembleModel(model_roberta, model_koelectra).to(device)

# 옵티마이저 초기화
optimizer = Adam(ensemble_model.parameters(), lr=1e-5)

# DataLoader 생성
batch_size = 16  # 원하는 배치 크기로 설정하세요
train_dataloader_roberta = DataLoader(dataset_roberta, batch_size=batch_size, shuffle=True)
train_dataloader_koelectra = DataLoader(dataset_koelectra, batch_size=batch_size, shuffle=True)


def loss_fn(outputs, labels):
    return F.cross_entropy(outputs, labels)


# Training loop
for epoch in range(5):
    ensemble_model.train()

    for batch_roberta, batch_koelectra in zip(train_dataloader_roberta, train_dataloader_koelectra):
        texts = tokenizer_roberta.batch_decode(batch_roberta['input_ids'], skip_special_tokens=True)
        labels = batch_roberta['labels'].to(device)

        optimizer.zero_grad()

        # Calculate model predictions
        final_labels, logits = ensemble_model(texts, tokenizer_roberta, tokenizer_koelectra, device)

        # Calculate the loss
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/50, Loss: {loss.item()}")


# 예측 함수
def predict_with_ensemble_modified(texts, ensemble_model, tokenizer_roberta, tokenizer_koelectra, device):
    ensemble_model.eval()
    with torch.no_grad():
        final_labels = ensemble_model(texts, tokenizer_roberta, tokenizer_koelectra, device)

    # star1과 star2 값을 출력
    print("star1 value:", ensemble_model.star1.item())
    print("star2 value:", ensemble_model.star2.item())

    return final_labels


# 사용 예제
texts = ["여기에 분석할 텍스트를 입력하세요."]
final_labels = predict_with_ensemble_modified(texts, ensemble_model, tokenizer_roberta, tokenizer_koelectra, device)
print("최종 레이블:", final_labels)