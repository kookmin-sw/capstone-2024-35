import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
# GPU 사용 가능 확인 및 device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 토크나이저 로딩
tokenizer_roberta = AutoTokenizer.from_pretrained("klue/roberta-large")
tokenizer_electra = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")


# 저장된 모델 로드
model_roberta = AutoModelForSequenceClassification.from_pretrained("Chamsol/klue-roberta-sentiment-classification")
model_electra = AutoModelForSequenceClassification.from_pretrained("Chamsol/koelctra-sentiment-classification")

#predict
def logits_to_probs(logits):
  return torch.nn.functional.softmax(logits, dim=1)

def predict_with_ensemble_modified(texts, roberta_model, koelectra_model, tokenizer_roberta, tokenizer_koelectra, device):
    encodings_roberta = tokenizer_roberta(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    encodings_koelectra = tokenizer_koelectra(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

    roberta_input_ids, roberta_attention_mask = encodings_roberta['input_ids'].to(device), encodings_roberta['attention_mask'].to(device)
    koelectra_input_ids, koelectra_attention_mask = encodings_koelectra['input_ids'].to(device), encodings_koelectra['attention_mask'].to(device)

    roberta_model.to(device)
    koelectra_model.to(device)

    roberta_model.eval()
    koelectra_model.eval()

    with torch.no_grad():
        roberta_outputs = roberta_model(roberta_input_ids, roberta_attention_mask)
        koelectra_outputs = koelectra_model(koelectra_input_ids, koelectra_attention_mask)

        roberta_probs = logits_to_probs(roberta_outputs.logits).cpu().numpy()
        koelectra_probs = logits_to_probs(koelectra_outputs.logits).cpu().numpy()

        ensemble_probs = (roberta_probs + koelectra_probs) / 2

        final_labels = []

        for probs in ensemble_probs:
            pred_label = np.argmax(probs)
            if pred_label == 1 and probs[pred_label] > 0.88:  # 긍정이면서 확률이 0.8 이상인 경우
                pred_label = 3  # 매우 긍정으로 변경
            elif pred_label == 0 and probs[pred_label] > 0.86:  # 부정이면서 확률이 0.8 이상인 경우
                pred_label = 2  # 매우 부정으로 변경
            final_labels.append(pred_label)

    return final_labels