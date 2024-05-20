#pre_train.py
#ec2-train

from transformers import AutoModelForSequenceClassification
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from transformers import EvalPrediction
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
import pandas as pd


df = pd.read_csv('/home/ec2-user/train_org.csv')  # 'your_dataset.csv'를 실제 파일 경로로 변경하세요.


# 데이터 전처리
texts = df['document'].tolist()  # 리뷰 텍스트
labels = df['label'].tolist()  # 라벨 데이터는 정수로 변환되어 있어야 합니다..
labels = [int(label) for label in labels]  # 라벨을 정수형으로 변환
texts = [str(text) for text in texts if text is not None]

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
roberta_model_path = "/home/ec2-user/roberta_model"
# 로드할 Electra 모델 경로
electra_model_path = "/home/ec2-user/electra_model"

# 저장된 모델 로드
model_roberta = AutoModelForSequenceClassification.from_pretrained(roberta_model_path)
model_electra = AutoModelForSequenceClassification.from_pretrained(electra_model_path)


# 사용자 지정 콜백 클래스 정의
class PrintProgressCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch} 종료. 진행 상태: {state.global_step}/{state.max_steps} ({(state.global_step/state.max_steps)*100:.2f}%)")
        if 'eval_metrics' in kwargs:
            print(f"Epoch {state.epoch} Accuracy: {kwargs['eval_metrics']['eval_accuracy']:.4f}")

# 위에서 정의한 compute_metrics 함수 사용
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}


class CustomTrainer(Trainer):
    def __init__(self, label_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_weights = label_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # 모델의 출력에서 로스 계산
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # 가중치 적용하여 로스 계산
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.label_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

   # def on_step_end(self, args, state, control, **kwargs):
   #     super().on_step_end(args, state, control, **kwargs)
   #     # 매 1000 스텝마다 모델 저장
   #     if state.global_step % 10 == 0:
   #         output_dir1 = f"{args.output_dir1}/checkpoint-{state.global_step}"
   #         self.save_model(output_dir1)
   #         print(f"모델을 {output_dir1}에 저장했습니다.")
   #     if state.global_step % 10 == 0:
   #         output_dir2 = f"{args.output_dir2}/checkpoint-{state.global_step}"
   #         self.save_model(output_dir2)
   #         print(f"모델을 {output_dir2}에 저장했습니다.")



# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir= './result',
    num_train_epochs=1,
    per_device_train_batch_size=32,
    warmup_steps=160000000,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=800,
)


# 라벨 가중치 설정
# 예시: 각 라벨에 대한 가중치가 동일하다고 가정
label_weights = torch.tensor([1.0, 1.0, 1.0, 1.0]).to(device)

# CustomTrainer 인스턴스 생성 시 사용자 지정 콜백 추가
trainer_roberta = CustomTrainer(
    model=model_roberta,
    args=training_args,
    train_dataset=dataset_roberta,
    label_weights=label_weights,
    compute_metrics=compute_metrics,
    callbacks=[PrintProgressCallback()],  # 콜백 추가
)

trainer_electra = CustomTrainer(
    model=model_electra,
    args=training_args,
    train_dataset=dataset_electra,
    label_weights=label_weights,
    compute_metrics=compute_metrics,
    callbacks=[PrintProgressCallback()],  # 콜백 추가
)

# 훈련 시작
trainer_roberta.train()
trainer_electra.train()

# 훈련된 RoBERTa 모델 저장
model_save_path = "/home/ec2-user/to/save/roberta_model"
trainer_roberta.save_model(model_save_path)

# 훈련된 Electra 모델 저장
model_save_path = "/home/ec2-user/to/save/electra_model"
trainer_electra.save_model(model_save_path)