[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/omXkVCQu)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=14182909&assignment_repo_type=AssignmentRepo)
# 운동 영상 및 리뷰 사이트 SSAFIT
## 1.프로젝트 소개
![image](https://github.com/kookmin-sw/capstone-2024-35/assets/162407707/6b453b05-ba72-4bd6-8e4c-f8955ecee624)


## 2. 작업 순서
### 2-1 WBS 

### 2-2 FIGMA

### 2-3 ERD DIAGRAM

## 3. 팀 소개
<table>
    <tr align="center">
        <td><img src=""
 width="250"></td>
        <td><img src=""
 width="250"></td>
        <td><img src="
 width="250"></td>
    </tr>
    <tr align="center">
        <td>이동현</td>
        <td>김동욱</td>
    </tr>
    <tr align="center">
        <td>Full stack</td>
        <td>Full stack</td>
    </tr>
</table>
<br>



## 4. 🛠 기술 스택

### 🌐 Frontend
- ![JavaScript](https://img.shields.io/badge/-JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black) - The programming language of the Web.
- ![REACT](https://img.shields.io/badge/-React-61DAFB?style=flat&logo=react&logoColor=black) - A JavaScript library for building user interfaces.
- ![Figma](https://img.shields.io/badge/-Figma-F24E1E?style=flat&logo=figma&logoColor=white) - Utilized for designing the overall UI/UX of the frontend.


### ⚙️ Backend
- ![NodeJS](https://img.shields.io/badge/-NodeJS-339933?style=flat&logo=nodedotjs&logoColor=white) - JavaScript runtime built on Chrome's V8 JavaScript engine.
- ![Express](https://img.shields.io/badge/-Express-000000?style=flat&logo=express&logoColor=white) - Fast, unopinionated, minimalist web framework for Node.js.
- ![MongoDB](https://img.shields.io/badge/-MongoDB-47A248?style=flat&logo=mongodb&logoColor=white) - NoSQL database that uses JSON-like documents with schemata.

### 🤖 AI
- ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white) - High-level programming language for general-purpose programming.
- ![Pytorch](https://img.shields.io/badge/-Pytorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) - An open source machine learning library based on the Torch library.
- ![Colab](https://img.shields.io/badge/-Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white) - A Google research project created to help disseminate machine learning education and research.
- ![AWS EC2](https://img.shields.io/badge/-AWS%20EC2-232F3E?style=flat&logo=amazonaws&logoColor=white) - Utilizing GPU instances for high-performance computing in machine learning projects.


### 🧰 Common Tools
- ![Github](https://img.shields.io/badge/-Github-181717?style=flat&logo=github&logoColor=white) - Provides hosting for software development version control using Git.
- ![Notion](https://img.shields.io/badge/-Notion-000000?style=flat&logo=notion&logoColor=white) - An application that provides components such as notes, databases, kanban boards, wikis, calendars and reminders.




## 5. 서비스 구조도



## 6. 사용방법
# 프로젝트명

## 개요
이 프로젝트는 Java와 Mysql를 사용하여 개발되었습니다.

## 환경 설정
- Node.js v20.11.1(LTS) 설치
- 로컬에 MongoDB 설치

### Node.js 설치
1. Node.js v20.11.1(LTS)를 설치합니다. [Node.js 다운로드 페이지](https://nodejs.org/)에서 다운로드 및 설치할 수 있습니다.

### 로컬 MongoDB 설치
1. MongoDB 공식 사이트에서 MongoDB Community Server를 다운로드하고 설치합니다. [MongoDB 다운로드 페이지](https://www.mongodb.com/try/download/community)

### 모델 배포
1. 모델은 huggingface에 업로드 되어있습니다 . [모델 주소 : https://huggingface.co/mmoonssun/klue_ner_kobert]
2. [klue-roberta_model: https://huggingface.co/Chamsol/klue-roberta-sentiment-classification/tree/main]
3. [koelectra_model: https://huggingface.co/Chamsol/klue-roberta-sentiment-classification]
4. 파이썬이나 colab등에서 모델을 불러오려면 다음 코드를 실행하면 됩니다.
```python
from kobert_transformers import get_kobert_model, get_tokenizer
from transformers import BertForTokenClassification

model_name = "mmoonssun/klue_ner_kobert"
model = BertForTokenClassification.from_pretrained(model_name, num_labels=13)
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
```

## 패키지 설치
프로젝트에 필요한 패키지들을 설치합니다.
npm install

## 추후 변경해야 할 점






## 7. 캡스톤 최종발료 자료 
https://docs.google.com/presentation/d/1iJvgJuWEdSxu9rag-1UDNdaGIv3s8Zqh/edit?usp=sharing&ouid=116159848948864038515&rtpof=true&sd=true

## 8. 캡스톤 포스터 ai 파일
https://drive.google.com/file/d/1brOpfNPwoGO98SUKq56ew-wEG-nqXEWk/view?usp=drive_link
