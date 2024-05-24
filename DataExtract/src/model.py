from transformers import BertForTokenClassification
from kobert_tokenizer import KoBERTTokenizer
import torch
import re

def load_model_and_tokenizer():
    model_name = "mmoonssun/klue_ner_kobert"
    model = BertForTokenClassification.from_pretrained(model_name, num_labels=13)
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    return model, tokenizer
def preprocess_text(text):
    # 숫자 앞뒤에 공백 추가
    text = re.sub(r'(\d+)', r' \1 ', text)
    text = text.replace('/', ' , ')
    # 중복 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def predict_entities(text, model, tokenizer, id2tag):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    text = preprocess_text(text)  # 텍스트 전처리 적용
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)
    predicted_tags = [id2tag[id.item()] for id in predictions[0]]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    token_tag_pairs = [(token.replace('▁', ' '), tag) for token, tag in zip(tokens, predicted_tags) if token not in ["[CLS]", "[SEP]", "[PAD]", "<pad>"]]
    print(token_tag_pairs)
    return token_tag_pairs
