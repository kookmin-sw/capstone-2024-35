import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# GPU 사용 가능 확인 및 device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 토크나이저 로딩
tokenizer_roberta = AutoTokenizer.from_pretrained("klue/roberta-large")
tokenizer_electra = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

# 로드할 RoBERTa 모델 경로
roberta_model_path = "/content/drive/MyDrive/to/save/roberta_model"
# 로드할 Electra 모델 경로
electra_model_path = "/content/drive/MyDrive/to/save/electra_model"

# 저장된 모델 로드
model_roberta = AutoModelForSequenceClassification.from_pretrained(roberta_model_path)
model_electra = AutoModelForSequenceClassification.from_pretrained(electra_model_path)
# predict_with_ensemble_modified
# 소프트맥스 함수를 사용하여 로짓을 확률로 변환하는 함수
def logits_to_probs(logits):
    return torch.nn.functional.softmax(logits, dim=1)

def predict_with_ensemble_modified(texts, roberta_model, koelectra_model, tokenizer_roberta, tokenizer_koelectra, device):
    encodings_roberta = tokenizer_roberta(texts, truncation=True, padding=True, max_length=128)
    encodings_koelectra = tokenizer_koelectra(texts, truncation=True, padding=True, max_length=128)
    roberta_dataset = SentimentDataset(encodings_roberta, [0]*len(texts))
    koelectra_dataset = SentimentDataset(encodings_koelectra, [0]*len(texts))

    roberta_dataloader = DataLoader(roberta_dataset, batch_size=32, shuffle=False)
    koelectra_dataloader = DataLoader(koelectra_dataset, batch_size=32, shuffle=False)

    roberta_model.to(device)
    koelectra_model.to(device)

    roberta_model.eval()
    koelectra_model.eval()

    final_labels = []

    with torch.no_grad():
        for roberta_batch, koelectra_batch in zip(roberta_dataloader, koelectra_dataloader):
            roberta_input_ids, roberta_attention_mask = roberta_batch['input_ids'].to(device), roberta_batch['attention_mask'].to(device)
            koelectra_input_ids, koelectra_attention_mask = koelectra_batch['input_ids'].to(device), koelectra_batch['attention_mask'].to(device)

            roberta_outputs = roberta_model(roberta_input_ids, roberta_attention_mask)
            koelectra_outputs = koelectra_model(koelectra_input_ids, koelectra_attention_mask)

            roberta_probs = logits_to_probs(roberta_outputs.logits).cpu().numpy()
            koelectra_probs = logits_to_probs(koelectra_outputs.logits).cpu().numpy()

            ensemble_probs = (roberta_probs + koelectra_probs) / 2

            final_labels = []

            for probs in ensemble_probs:
                pred_label = np.argmax(probs)
                if pred_label == 1 and probs[pred_label] > 0.6:  # 긍정이면서 확률이 0.6 이상인 경우
                    pred_label = 3  # 매우 긍정으로 변경
                elif pred_label == 0 and probs[pred_label] > 0.7:  # 부정이면서 확률이 0.7 이상인 경우
                    pred_label = 2  # 매우 부정으로 변경
                final_labels.append(pred_label)  # 수정된 라벨을 최종 라벨 리스트에 추가

    return final_labels  # 수정: 최종 라벨 리스트 반환

# 예측할 텍스트 시퀀스
texts_to_predict = ["일은 보통으로 하고 사람은 좋음.","일도 매우 잘하고 사람도 좋고 성실함", "일은 잘못하지만 말은 잘 들음", "불성실하고 매우 필요없음 그냥 없는게 나음"]

# 함수 호출에 device 변수 사용
final_labels = predict_with_ensemble_modified(texts_to_predict, model_roberta, model_electra, tokenizer_roberta, tokenizer_electra, device)

# 결과 출력
for text, label in zip(texts_to_predict, final_labels):
    print(f"Text: {text} - Prediction: {label}")
