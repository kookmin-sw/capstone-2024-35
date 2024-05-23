#get label
import pandas as pd
import os
import pandas as pd
from tqdm import tqdm
import openai


def get_completion(prompt, model="gpt-3.5-turbo", api_key=""):
    openai.api_key = api_key
    messages = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )

    return response.choices[0].message["content"]

def get_label(person_name, person_corpus: str, api_key: str) -> str:
    prompt = f'''### 역할:
                다음 설명을 기반으로 {person_name}에 대한 평가가 "매우 긍정", "긍정", "부정", "매우 부정" 중 어느 것에 해당하는지 라벨링 하시오. 이는 아래 규칙을 따릅니다.

                ### 설명:
                {person_corpus}

                ### 규칙:
                1. 주어진 설명에 대해 하나의 ["매우 긍정"은 토큰값으로 3, "긍정"은 토큰값으로 1, "부정"은 토큰값으로 0, "매우 부정"은 토큰값으로 2]만을 선택하여 라벨링한다.
                2. 설명은 "### 설명:" 부분부터 시작된다.
                3. 출력은 "### 출력"와 같이 json 형식으로 출력한다.
                4. 라벨링은 설명 전문에 대해서 한다.
                5. 중립은 선택지에 없다.
                6. 출력은 형식외에 아무것도 작성하지 않는다.

                ### 출력
                '''

    response = get_completion(prompt, "gpt-3.5-turbo", api_key)
    return response

if __name__ == "__main__":
    path_file = "path"  # 데이터셋 파일 경로 지정
    api_key = "OpenAi api key"  # OpenAI API 키 지정
    pos_start = 0
    len_slice = 500



    df_people = pd.read_csv(path_file)[pos_start:pos_start+len_slice]
    df_people = df_people.drop_duplicates(["person_name"])
    df_people = df_people.dropna()

    list_result = []

    for idx, row in tqdm(df_people.iterrows(), total=df_people.shape[0]):
        try:
            label_response = get_label(row["person_name"], row["person_corpus"], api_key)
            list_result.append(label_response)
        except Exception as e:
            print(f"idx: {idx}, err: {e}")
            list_result.append(f"idx: {idx}, err: {e}")

    df_people["label"] = list_result

    df_people.to_csv(f"./labeled_people_{pos_start}_to_{pos_start + len_slice}.csv", index=False)