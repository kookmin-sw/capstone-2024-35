from transformers import BertForTokenClassification
from kobert_tokenizer import KoBERTTokenizer
import torch

def load_model_and_tokenizer():
    model_name = "mmoonssun/klue_ner_kobert"
    model = BertForTokenClassification.from_pretrained(model_name, num_labels=13)
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    return model, tokenizer

def predict_entities(text, model, tokenizer, id2tag):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
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
    return token_tag_pairs
