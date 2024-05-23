import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from sklearn.metrics import accuracy_score
from transformers import TrainerCallback


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
df = pd.read_csv('path')  # hr_data
# 데이터 전처리
texts = df['person_corpus'].tolist()
labels = df['label'].tolist()  # 라벨 데이터는 정수로 변환되어 있어야 합니다.

# 토크나이저 로딩
tokenizer_roberta = AutoTokenizer.from_pretrained("klue/roberta-large")
tokenizer_electra = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

# 데이터셋 토큰화
encodings_roberta = tokenizer_roberta(texts, truncation=True, padding=True, max_length=128)
encodings_electra = tokenizer_electra(texts, truncation=True, padding=True, max_length=128)

# 데이터셋 생성
dataset_roberta = SentimentDataset(encodings_roberta, labels)
dataset_electra = SentimentDataset(encodings_electra, labels)


# 로드할 RoBERTa 모델 경로
roberta_model_path = "path"
# 로드할 Electra 모델 경로
electra_model_path = "path"

# 저장된 모델 로드
model_roberta = AutoModelForSequenceClassification.from_pretrained(roberta_model_path)
model_electra = AutoModelForSequenceClassification.from_pretrained(electra_model_path)

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
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# 라벨별 가중치 설정
label_weights = {0: 8.0, 1: 2.5, 2: 5.0, 3: 1.0}

# CustomTrainer 인스턴스 생성
trainer_roberta = CustomTrainer(
    model=model_roberta,
    args=training_args,
    train_dataset=dataset_roberta,  # dataset_roberta는 사용자가 정의해야 함
    label_weights=label_weights,
)

trainer_electra = CustomTrainer(
    model=model_electra,
    args=training_args,
    train_dataset=dataset_electra,  # dataset_electra는 사용자가 정의해야 함
    label_weights=label_weights,
)

# 로짓을 확률로 변환하는 함수
def logits_to_probs(logits):
    return torch.softmax(logits, dim=1)

def predict_with_ensemble_modified(texts, roberta_model, koelectra_model, tokenizer_roberta, tokenizer_koelectra, device):
    encodings_roberta = tokenizer_roberta(texts, truncation=True, padding=True, max_length=128)
    encodings_koelectra = tokenizer_koelectra(texts, truncation=True, padding=True, max_length=128)

    roberta_dataset = SentimentDataset(encodings_roberta, [0]*len(texts))  # 라벨은 예측을 위해 사용되지 않으므로 임시 값으로 설정
    koelectra_dataset = SentimentDataset(encodings_koelectra, [0]*len(texts))

    roberta_dataloader = DataLoader(roberta_dataset, batch_size=32, shuffle=False)
    koelectra_dataloader = DataLoader(koelectra_dataset, batch_size=32, shuffle=False)

    roberta_model.to(device)
    koelectra_model.to(device)

    roberta_model.eval()
    koelectra_model.eval()

    final_labels = []  # 최종 라벨을 저장할 리스트를 루프 외부에서 초기화

    with torch.no_grad():
        for roberta_batch, koelectra_batch in zip(roberta_dataloader, koelectra_dataloader):
            roberta_input_ids, roberta_attention_mask = roberta_batch['input_ids'].to(device), roberta_batch['attention_mask'].to(device)
            koelectra_input_ids, koelectra_attention_mask = koelectra_batch['input_ids'].to(device), koelectra_batch['attention_mask'].to(device)

            roberta_outputs = roberta_model(roberta_input_ids, roberta_attention_mask)
            koelectra_outputs = koelectra_model(koelectra_input_ids, koelectra_attention_mask)

            roberta_probs = logits_to_probs(roberta_outputs.logits).cpu().numpy()
            koelectra_probs = logits_to_probs(koelectra_outputs.logits).cpu().numpy()

            ensemble_probs = (roberta_probs + koelectra_probs) / 2

            for probs in ensemble_probs:
                pred_label = np.argmax(probs)
                if pred_label == 1 and probs[pred_label] > 0.6:  # 긍정이면서 확률이 0.5 이상인 경우
                    pred_label = 3  # 매우 긍정으로 변경
                elif pred_label == 0 and probs[pred_label] > 0.7:  # 부정이면서 확률이 0.5 이상인 경우
                    pred_label = 2  # 매우 부정으로 변경
                final_labels.append(pred_label)  # 수정된 라벨을 최종 라벨 리스트에 추가

    return final_labels  # 수정: 최종 라벨 리스트 반환

# 함수 호출에 device 변수 사용
final_labels = predict_with_ensemble_modified(texts, model_roberta, model_electra, tokenizer_roberta, tokenizer_electra, device)

# 결과 출력
for text, label in zip(texts, final_labels):
    print(f"Text: {text} - Prediction: {label}")
