from geopy.distance import geodesic
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# GPU 사용 가능 확인 및 device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 토크나이저 로딩
tokenizer_roberta = AutoTokenizer.from_pretrained("klue/roberta-large")
tokenizer_electra = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

# 저장된 모델 로드
model_roberta = AutoModelForSequenceClassification.from_pretrained("Chamsol/klue-roberta-sentiment-classification")
model_electra = AutoModelForSequenceClassification.from_pretrained("Chamsol/koelctra-sentiment-classification")


def calculate_distance(point1, point2):
    # 두 지점 사이의 거리를 계산하는 함수
    distance_to_station = geodesic(point1, point2).kilometers
    return distance_to_station



def label_to_value(label):
    if label == 0:
        return -30
    elif label == 1:
        return 30
    elif label == 2:
        return -50
    elif label == 3:
        return 50
    else:
        return 0

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
            if pred_label == 1 and probs[pred_label] > 0.82:  # 긍정이면서 확률이 0.8 이상인 경우
                pred_label = 3  # 매우 긍정으로 변경
            elif pred_label == 0 and probs[pred_label] > 0.89:  # 부정이면서 확률이 0.8 이상인 경우
                pred_label = 2  # 매우 부정으로 변경
            final_labels.append(pred_label)

    return final_labels

def get_coordinates_worksites(address):
    # 주소에 따른 좌표 정보를 저장하는 딕셔너리
    coordinates_worksites = {
        "서울": {'work_location_x': 37.5647, 'work_location_y': 126.9770},
        "강남구": {'work_location_x': 37.4979, 'work_location_y': 127.0276},
        "강동구": {'work_location_x': 37.5385, 'work_location_y': 127.1238},
        "강북구": {'work_location_x': 37.6380, 'work_location_y': 127.0251},
        "강서구": {'work_location_x': 37.5316, 'work_location_y': 126.8467},
        "관악구": {'work_location_x': 37.4813, 'work_location_y': 126.9527},
        "광진구": {'work_location_x': 37.5351, 'work_location_y': 127.0857},
        "구로구": {'work_location_x': 37.4853, 'work_location_y': 126.9015},
        "금천구": {'work_location_x': 37.4665, 'work_location_y': 126.8893},
        "노원구": {'work_location_x': 37.6445, 'work_location_y': 127.0643},
        "도봉구": {'work_location_x': 37.6531, 'work_location_y': 127.0476},
        "동대문구": {'work_location_x': 37.5716, 'work_location_y': 127.0095},
        "동작구": {'work_location_x': 37.5082, 'work_location_y': 126.9637},
        "마포구": {'work_location_x': 37.5572, 'work_location_y': 126.9236},
        "서대문구": {'work_location_x': 37.5658, 'work_location_y': 126.9666},
        "서초구": {'work_location_x': 37.4919, 'work_location_y': 127.0072},
        "성동구": {'work_location_x': 37.5612, 'work_location_y': 127.0379},
        "성북구": {'work_location_x': 37.5926, 'work_location_y': 127.0163},
        "송파구": {'work_location_x': 37.5133, 'work_location_y': 127.1001},
        "양천구": {'work_location_x': 37.5263, 'work_location_y': 126.8643},
        "영등포구": {'work_location_x': 37.5171, 'work_location_y': 126.9173},
        "용산구": {'work_location_x': 37.5298, 'work_location_y': 126.9644},
        "은평구": {'work_location_x': 37.6190, 'work_location_y': 126.9215},
        "종로구": {'work_location_x': 37.5704, 'work_location_y': 126.9910},
        "중구": {'work_location_x': 37.5647, 'work_location_y': 126.9770},
        "중랑구": {'work_location_x': 37.5960, 'work_location_y': 127.085},
        "인천": {'work_location_x': 37.5715, 'work_location_y': 126.7354},
        "인천 강화군": {'work_location_x': 37.5692, 'work_location_y': 126.6732},
        "인천 계양구": {'work_location_x': 37.5715, 'work_location_y': 126.7354},
        "인천 남동구": {'work_location_x': 37.4488, 'work_location_y': 126.7013},
        "인천 동구": {'work_location_x': 37.4751, 'work_location_y': 126.6322},
        "인천 미추홀구": {'work_location_x': 37.4420, 'work_location_y': 126.6995},
        "인천 부평구": {'work_location_x': 37.4891, 'work_location_y': 126.7245},
        "인천 서구": {'work_location_x': 37.5692, 'work_location_y': 126.6732},
        "인천 연수구": {'work_location_x': 37.3866, 'work_location_y': 126.6392},
        "인천 옹진군": {'work_location_x': 37.5111, 'work_location_y': 126.5230},
        "인천 중구": {'work_location_x': 37.4764, 'work_location_y': 126.6169},
        "가평군": {'가평역': {'work_location_x': 37.8147, 'work_location_y': 127.5102}},
        "고양시 일산서구": {'work_location_x': 37.6580, 'work_location_y': 126.7942},
        "고양시 일산동구": {'work_location_x': 37.6586, 'work_location_y': 126.7698},
        "고양시 덕양구": {'work_location_x': 37.6347, 'work_location_y': 126.8328},
        "과천시": {'work_location_x': 37.4339, 'work_location_y': 126.9966},
        "광명시": {'work_location_x': 37.4163, 'work_location_y': 126.8840},
        "광주시": {'work_location_x': 37.4095, 'work_location_y': 127.2550},
        "구리시": {'work_location_x': 37.6038, 'work_location_y': 127.1436},
        "군포시": {'work_location_x': 37.3532, 'work_location_y': 126.9488},
          "김포시": {"work_location_x": 37.5613, "work_location_y": 126.8019},
  "남양주시": {"work_location_x": 37.6423, "work_location_y": 127.1264},
  "동두천시": {"work_location_x": 37.8922, "work_location_y": 127.0603},
  "부천시 소사구": {"subway_location_x": 37.4827, "subway_location_y": 126.7950},
"부천시 원미구": {"subway_location_x": 37.5047, "subway_location_y": 126.7630},
"부천시 오정구": {"subway_location_x": 37.5142, "subway_location_y": 126.7928},
  "성남시 수정구": {"work_location_x": 37.4519, "work_location_y": 127.1584},
  "성남시 분당구": {"work_location_x": 37.3595, "work_location_y": 127.1086},
  "성남시 중원구": {"work_location_x": 37.4321, "work_location_y": 127.1150},
  "수원시 장안구": {"work_location_x": 37.2986, "work_location_y": 127.0107},
  "수원시 권선구": {"work_location_x": 37.2699, "work_location_y": 127.0286},
  "수원시 팔달구": {"work_location_x": 37.2657, "work_location_y": 126.9996},
  "수원시 영통구": {"work_location_x": 37.2886, "work_location_y": 127.0511},
  "시흥시": {"work_location_x": 37.3800, "work_location_y": 126.8035},
  "안산시 단원구": {"work_location_x": 37.3180, "work_location_y": 126.8386},
        "안산시 상록구": {"work_location_x": 37.3083, "work_location_y": 126.8530},

  "안성시": {"work_location_x": 37.0100, "work_location_y": 127.2701},
  "안양시 만안구": {"work_location_x": 37.4352, "work_location_y": 126.9021},
  "안양시 동안구": {"work_location_x": 37.0100, "work_location_y": 127.2701},
  "양주시": {"work_location_x": 37.7840, "work_location_y": 127.0457},
  "양평군": {"work_location_x": 37.4910, "work_location_y": 127.4874},
  "여주시": {"work_location_x": 37.2822, "work_location_y": 127.6280},
  "연천군": {"work_location_x": 38.0960, "work_location_y": 127.0741},
  "오산시": {"work_location_x": 37.1451, "work_location_y": 127.0660},
  "용인시 처인구": {"work_location_x": 37.2821, "work_location_y": 127.1216},
  "용인시 기흥구": {"work_location_x": 37.2749, "work_location_y": 127.1150},
  "용인시 수지구": {"work_location_x": 37.3377, "work_location_y": 127.0988},
  "의왕시": {"work_location_x": 37.3204, "work_location_y": 126.9483},
  "의정부시": {"work_location_x": 37.7377, "work_location_y": 127.0474},
  "이천시": {"work_location_x": 37.2725, "work_location_y": 127.4348},
  "파주시": {"work_location_x": 37.7126, "work_location_y": 126.7610},
  "평택시": {"work_location_x": 36.9908, "work_location_y": 127.0856},
  "포천시": {"work_location_x": 37.8945, "work_location_y": 127.2006},
  "하남시": {"work_location_x": 37.5511, "work_location_y": 127.2060},
  "화성시": {"work_location_x": 37.1988, "work_location_y": 127.1034}

    }

    # 주어진 주소의 좌표 정보를 추출
    coordinates = coordinates_worksites.get(address)
    if coordinates is not None:
        # 좌표 정보가 있으면 x, y 좌표만을 튜플 형태로 반환
        return (coordinates["work_location_x"], coordinates["work_location_y"])
    else:
        # 주소에 해당하는 좌표 정보가 없으면 None 반환
        return None

    # 주어진 주소에 따라 좌표 반환
    return coordinates_worksites.get(address)  # 주소가 없으면 None 반환

def get_coordinates_employee(address):
    # 주소에 따른 좌표 정보를 저장하는 딕셔너리
    coordinates_employee = {
        "서울": {"subway_location_x": 37.5647, "subway_location_y": 126.9770},
  "강남구": {"subway_location_x": 37.4979, "subway_location_y": 127.0276},
  "강동구": {"subway_location_x": 37.5385, "subway_location_y": 127.1238},
  "강북구": {"subway_location_x": 37.6380, "subway_location_y": 127.0251},
  "강서구": {"subway_location_x": 37.5316, "subway_location_y": 126.8467},
  "관악구": {"subway_location_x": 37.4813, "subway_location_y": 126.9527},
  "광진구": {"subway_location_x": 37.5351, "subway_location_y": 127.0857},
  "구로구": {"subway_location_x": 37.4853, "subway_location_y": 126.9015},
  "금천구": {"subway_location_x": 37.4665, "subway_location_y": 126.8893},
  "노원구": {"subway_location_x": 37.6445, "subway_location_y": 127.0643},
  "도봉구": {"subway_location_x": 37.6531, "subway_location_y": 127.0476},
  "동대문구": {"subway_location_x": 37.5716, "subway_location_y": 127.0095},
  "동작구": {"subway_location_x": 37.5082, "subway_location_y": 126.9637},
  "마포구": {"subway_location_x": 37.5572, "subway_location_y": 126.9236},
  "서대문구": {"subway_location_x": 37.5658, "subway_location_y": 126.9666},
  "서초구": {"subway_location_x": 37.4919, "subway_location_y": 127.0072},
  "성동구": {"subway_location_x": 37.5612, "subway_location_y": 127.0379},
  "성북구": {"subway_location_x": 37.5926, "subway_location_y": 127.0163},
  "송파구": {"subway_location_x": 37.5133, "subway_location_y": 127.1001},
  "양천구": {"subway_location_x": 37.5263, "subway_location_y": 126.8643},
  "영등포구": {"subway_location_x": 37.5171, "subway_location_y": 126.9173},
  "용산구": {"subway_location_x": 37.5298, "subway_location_y": 126.9644},
  "은평구": {"subway_location_x": 37.6190, "subway_location_y": 126.9215},
  "종로구": {"subway_location_x": 37.5704, "subway_location_y": 126.9910},
  "중구": {"subway_location_x": 37.5647, "subway_location_y": 126.9770},
  "중랑구": {"subway_location_x": 37.5960, "subway_location_y": 127.085},
  "인천": {"subway_location_x": 37.5715, "subway_location_y": 126.7354},
  "인천 강화군": {"subway_location_x": 37.5692, "subway_location_y": 126.6732},
  "인천 계양구": {"subway_location_x": 37.5715, "subway_location_y": 126.7354},
  "인천남동구": {"subway_location_x": 37.4488, "subway_location_y": 126.7013},
  "인천 동구": {"subway_location_x": 37.4751, "subway_location_y": 126.6322},
  "인천 미추홀구": {"subway_location_x": 37.4420, "subway_location_y": 126.6995},
  "인천 부평구": {"subway_location_x": 37.4891, "subway_location_y": 126.7245},
  "인천 서구": {"subway_location_x": 37.5692, "subway_location_y": 126.6732},
  "인천 연수구": {"subway_location_x": 37.3866, "subway_location_y": 126.6392},
  "인천 옹진군": {"subway_location_x": 37.5111, "subway_location_y": 126.5230},
  "인천 중구": {"subway_location_x": 37.4764, "subway_location_y": 126.6169},
  "가평군": {"subway_location_x": 37.8147, "subway_location_y": 127.5102},
  "고양시 일산서구": {"subway_location_x": 37.6580, "subway_location_y": 126.7942},
  "고양시 일산동구": {"subway_location_x": 37.6586, "subway_location_y": 126.7698},
  "고양시 덕양구": {"subway_location_x": 37.6347, "subway_location_y": 126.8328},
  "과천시": {"subway_location_x": 37.4339, "subway_location_y": 126.9966},
  "광명시": {"subway_location_x": 37.4163, "subway_location_y": 126.8840},
  "광주시": {"subway_location_x": 37.4095, "subway_location_y": 127.2550},
  "구리시": {"subway_location_x": 37.6038, "subway_location_y": 127.1436},
  "군포시": {"subway_location_x": 37.3532, "subway_location_y": 126.9488},
  "김포시": {"subway_location_x": 37.5613, "subway_location_y": 126.8019},
  "남양주시": {"subway_location_x": 37.6423, "subway_location_y": 127.1264},
  "동두천시": {"subway_location_x": 37.8922, "subway_location_y": 127.0603},
  "부천시 소사구": {"subway_location_x": 37.4827, "subway_location_y": 126.7950},
"부천시 원미구": {"subway_location_x": 37.5047, "subway_location_y": 126.7630},
"부천시 오정구": {"subway_location_x": 37.5142, "subway_location_y": 126.7928},

  "성남시 수정구": {"subway_location_x": 37.4519, "subway_location_y": 127.1584},
  "성남시 분당구": {"subway_location_x": 37.3595, "subway_location_y": 127.1086},
  "성남시 중원구": {"subway_location_x": 37.4321, "subway_location_y": 127.1150},
  "수원시 장안구": {"subway_location_x": 37.2986, "subway_location_y": 127.0107},
  "수원시 권선구": {"subway_location_x": 37.2699, "subway_location_y": 127.0286},
  "수원시 팔달구": {"subway_location_x": 37.2657, "subway_location_y": 126.9996},
  "수원시 영통구": {"subway_location_x": 37.2886, "subway_location_y": 127.0511},
  "시흥시": {"subway_location_x": 37.3800, "subway_location_y": 126.8035},
  "안산시 단원구": {"subway_location_x": 37.3180, "subway_location_y": 126.8386},
        "안산시 상록구": {"subway_location_x": 37.3083, "subway_location_y": 126.8530},

  "안성시": {"subway_location_x": 37.0100, "subway_location_y": 127.2701},
  "안양시 만안구": {"subway_location_x": 37.4352, "subway_location_y": 126.9021},
  "안양시 동안구": {"subway_location_x": 37.0100, "subway_location_y": 127.2701},
  "양주시": {"subway_location_x": 37.7840, "subway_location_y": 127.0457},
  "양평군": {"subway_location_x": 37.4910, "subway_location_y": 127.4874},
  "여주시": {"subway_location_x": 37.2822, "subway_location_y": 127.6280},
  "연천군": {"subway_location_x": 38.0960, "subway_location_y": 127.0741},
  "오산시": {"subway_location_x": 37.1451, "subway_location_y": 127.0660},
  "용인시 처인구": {"subway_location_x": 37.2821, "subway_location_y": 127.1216},
  "용인시 기흥구": {"subway_location_x": 37.2749, "subway_location_y": 127.1150},
  "용인시 수지구": {"subway_location_x": 37.3377, "subway_location_y": 127.0988},
  "의왕시": {"subway_location_x": 37.3204, "subway_location_y": 126.9483},
  "의정부시": {"subway_location_x": 37.7377, "subway_location_y": 127.0474},
  "이천시": {"subway_location_x": 37.2725, "subway_location_y": 127.4348},
  "파주시": {"subway_location_x": 37.7126, "subway_location_y": 126.7610},
  "평택시": {"subway_location_x": 36.9908, "subway_location_y": 127.0856},
  "포천시": {"subway_location_x": 37.8945, "subway_location_y": 127.2006},
  "하남시": {"subway_location_x": 37.5511, "subway_location_y": 127.2060},
  "화성시": {"subway_location_x": 37.1988, "subway_location_y": 127.1034}

  }
    # 주어진 주소의 좌표 정보를 추출
    coordinates = coordinates_employee.get(address)
    if coordinates is not None:
        # 좌표 정보가 있으면 x, y 좌표만을 튜플 형태로 반환
        return (coordinates["subway_location_x"], coordinates["subway_location_y"])
    else:
      #없으면 x,y좌표를 각각 50의 튜플로 반환
      return (50, 50)



# calculate_score_for_person 함수 내에서의 수정
def calculate_score_for_person(person_info, model_roberta, model_electra, tokenizer_roberta, tokenizer_electra, device):
    # 각 변수의 가중치
    weight_distance = 0.2
    weight_attendance = 0.3
    weight_work_frequency = 0.2
    weight_label_value = 0.3
    # 성별에 따라 가중치 조정
    gender_weight = 1 if person_info['sex'] == '남자' or person_info['sex'] == '여자' else 0

    work_location = get_coordinates_worksites(person_info['worksites_local'])  # 주소가 없으면 None 반환)
    subway_station = get_coordinates_employee(person_info['employee_local'])


    # 현장과 지하철역의 거리 계산 (km 단위로 환산)
    distance_to_station = calculate_distance(work_location, subway_station)


    # 출석률 계산
    actual_work_days = person_info['actual_work_days']
    applied_work_days = person_info['applied_work_days']
    attendance_rate = actual_work_days / applied_work_days * 100

    # 출석률에 따라 점수 할당
    if attendance_rate >= 100:
        attendance_score = 50
    elif attendance_rate >= 90:
        attendance_score = 20
    else:
        attendance_score = 0

    # 근무횟수에 따라 점수 할당
    if person_info['work_frequency'] >= 15:
        work_frequency_score = 50
    elif person_info['work_frequency'] >= 10:
        work_frequency_score = 30
    elif person_info['work_frequency'] >= 5:
        work_frequency_score = 20
    else:
        work_frequency_score = 0


    # 거리에 따라 점수 할당
    if distance_to_station < 10:
        distance_score = 50
    elif distance_to_station < 30:
        distance_score = 40
    elif distance_to_station < 50:
        distance_score = 30
    elif distance_to_station < 100:
        distance_score = 20
    elif distance_to_station < 200:
        distance_score = 10
    else:
        distance_score = 0

    # 텍스트 예측을 위한 코드 추가
    texts_to_predict = [person_info['review']]  # 텍스트를 리스트로 변환
    final_labels = predict_with_ensemble_modified(texts_to_predict, model_roberta, model_electra, tokenizer_roberta, tokenizer_electra, device)

    for label in final_labels:
      sentiment_score = label_to_value(label)
    print(final_labels)





    # 각 변수에 가중치를 곱하여 합산된 점수를 계산
    total_score = (distance_score * weight_distance) + \
                  (attendance_score * weight_attendance) + \
                  (work_frequency_score * weight_work_frequency) + \
                  (sentiment_score * weight_label_value)

    return total_score*gender_weight

# 예시 데이터베이스에서 각 사람들에 대한 정보 수정
people_info = [
    {'employee_local': "성북구", 'worksites_local': "성북구", 'sex': '남자', 'actual_work_days': 20, 'applied_work_days': 20, 'work_frequency': 20, 'review': "일은 보통으로 하고 사람은 좋음"},
    {'employee_local': '고양시 일산서구', 'worksites_local': '성북구', 'sex': '여자', 'actual_work_days': 10, 'applied_work_days': 10, 'work_frequency': 25, 'review': "일도 매우 잘하고 사람도 좋고 성실함"}, #업무 능숙도에 대한 corpus에 민감함
    {'employee_local': '고양시 일산서구', 'worksites_local': '성북구', 'sex': '여자', 'actual_work_days': 10, 'applied_work_days': 10, 'work_frequency': 25, 'review': "일은 잘못하고 불만은 없는 것 같음."},
    {'employee_local': '안산시 상록구', 'worksites_local': '동두천시', 'sex': '남자', 'actual_work_days': 10, 'applied_work_days': 10, 'work_frequency': 25, 'review': "불성실하고 매우 필요없음 그냥 없는게 나음"},

        {
        'employee_local': "성북구",
        'worksites_local': "성북구",
        'sex': '남자',
        'actual_work_days': 20,
        'applied_work_days': 20,
        'work_frequency': 20,
        'review': "김철수 과장은 업무 수행에 있어 매우 성실하고 책임감이 강한 직원입니다. 매일 정해진 시간에 출근하며, 주어진 과제를 항상 제 시간에 완수합니다. 특히, 그의 문제 해결 능력과 빠른 의사 결정 과정은 팀 내에서 큰 신뢰를 받고 있습니다. 김 과장은 고객 응대에서도 매우 친절하고 프로페셔널한 태도로 높은 만족도를 이끌어냈습니다.\n\n김 과장은 올해 여러 중요한 프로젝트를 성공적으로 이끌었습니다. 예를 들어, 지난 3월의 대규모 마케팅 캠페인에서는 기획부터 실행까지 주도적으로 참여하여 매출을 20% 증가시키는 성과를 거두었습니다. 그는 팀원들과의 협업에서도 뛰어난 리더십을 발휘하여 팀 전체의 사기를 높이고, 각 구성원의 강점을 최대한 이끌어내는 능력을 보여주었습니다.\n\n다만, 가끔 세부 사항에서의 꼼꼼함이 다소 부족할 때가 있어 추가적인 주의가 필요할 것으로 보입니다. 예를 들어, 보고서 작성 시 몇 차례 사소한 오탈자가 발견되었으며, 이는 김 과장이 업무의 세부 사항에 대한 관심을 더욱 기울일 필요가 있음을 시사합니다. 그러나 이는 김 과장의 전반적인 성과와 비교할 때 큰 문제가 되지는 않습니다.\n\n전반적으로 김철수 과장은 매우 우수한 업무 수행 능력과 태도를 가지고 있으며, 앞으로도 지속적인 성장이 기대되는 인재입니다. 그의 뛰어난 성과와 긍정적인 태도는 팀과 회사에 큰 자산이 될 것입니다. 따라서 김 과장은 앞으로도 중요한 프로젝트와 팀 내 주요 역할을 맡아 성공적으로 수행할 것으로 예상됩니다."
    }




]

#{'employee_local': '', 'worksites_local': '', 'sex': '', 'actual_work_days': , 'applied_work_days': , 'work_frequency': , 'review': ""}

# 각 사람들의 점수 계산
for person_info in people_info:
    score = calculate_score_for_person(person_info, model_roberta, model_electra, tokenizer_roberta, tokenizer_electra, device)
    print("총점:", score)