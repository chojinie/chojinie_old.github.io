---
layout: page
title: Attach the adjust Emoji
description: techeer bootcamp 2023
img: assets/img/emoji.gif
importance: 3
category: fun
---

When model explore image and detect human, then classifying the human emotion.
From this model match proper emoji with human emotion. It is very easy way because, (emotion, emoji) pair is memorized in DB.

# **IGE MOJI?**
<br>
 >AI가 얼굴 표정을 **분석**하여 선택한 이모지 템플릿에 맞는 **표정**을 사진에 **붙여넣어주는** 서비스 

</br>

## Main Flow
<p>
![team_e_IGEMOJI_Flow](https://user-images.githubusercontent.com/121562023/216638353-a20a1582-b256-42eb-bb41-6c6ae7be8e1e.jpg)
 - Main Feature : 원하는 이모지 템플릿 선택, 사용자가 업로드한 이미지속 사람의 표정을 AI가 분석하여 결과로 도출
 - Additional Feature : 원하는 이모지 템플릿을 생성 및 삭제 (다른 사용자들과의 공유)
</p>

## Features
- ***Main Page***

<img src = "https://user-images.githubusercontent.com/121562023/216717832-619db5a1-4ccc-4d78-a738-00027302e75e.gif"
 width="65%">
 
 - ***Login***
<img src = "https://user-images.githubusercontent.com/121562023/216719934-5baa2d26-f1f8-4b98-bcf9-ad90eac3b315.gif" width="65%">

- ***Emoji Select & Use***
<img src = "https://user-images.githubusercontent.com/121562023/216720145-5e9ce14e-88f3-46f3-a5b5-f53a80df756c.gif" width="65%">

- ***Emoji Create***
<img src = "https://user-images.githubusercontent.com/121562023/216720259-7d089a2b-81b2-4948-a12c-5649bf60cab5.gif" width="65%">

- ***Top3 Ranking***
<img src = "https://user-images.githubusercontent.com/121562023/216720434-69a55475-a946-4ad3-be0f-7cfb9d1b42b7.gif" width="65%">



## System Architecture

<img width="1200" alt="스크린샷 2023-02-01 오후 1 39 00" src="https://user-images.githubusercontent.com/121562023/216060082-0852cd9a-abbb-4dc6-80c1-31f0f4652fa4.png">

- **Frontend tech**
   - `MUI` : 인터랙티브 컴포넌트 기반의 React UI라이브러리
       - 높은 수준의 UI를 빠르고 효율적으로 개발할 수 있는 UI 도구
       
       - 리액트와 높은 호환성과 높은 수준의 퀄리티를 제공
<br></br>


   - `Redux` : Javascript application들의 state(상태)를 관리할 수 있는 라이브러리(저장소)
       - 손쉬운 state 관리
       
       - 편리한 웹사이트 상태 확인



- **Backend tech**

   - `RabbitMQ` : AMQP(Advanced Message Queuing Protocol)를 따르는 오픈소스 메세지 브로커
       - 유연한 라우팅 규칙 적용 가능
       - 메시지 전송 타이밍 제어(메시지 만료 또는 메시지 지연 제어) 
       - 단순하게 소비자 기능 구현 가능


   - `Celery` : 분산 메시지 전달을 기반으로 동작하는 비동기 작업 큐(Asynchronous Task/Job Queue)
          - 최적화 설정(RabbitMQ, librabbitmq 등)을 통해 비약적인 작업처리 제공
          - 확장성이 매우 뛰어나 거의 모든 부분을 커스텀하여 사용 가능( serializer, compression schemes, logging, schedulers, consumers, producers 등)


   - `Mainserver(Django)` 와 `AI-server(Flask)` 의 분리
        - MainServer(Django) : 하나의 프로젝트 내에 다수의 앱을 운영할 수 있는 ‘분할’의 특징이 있으며, 자체 ORM을 제공
        - AI-Server(Flask) : 무게가 가벼운 프레임워크
        - 각각의 프레임워크의 장점을 상황에 알맞게 사용

<br>

## **Tech Stack**
|Dev-Ops|<img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=Docker&logoColor=black"> <img src="https://img.shields.io/badge/Amazon EC2-FF9900?style=for-the-badge&logo=Amazon%20EC2&logoColor=white"> <img src="https://img.shields.io/badge/S3%20Bucket-569A31?style=for-the-badge&logo=Amazon%20S3&logoColor=white"> <img src="https://img.shields.io/badge/NGINX-009639?style=for-the-badge&logo=NGINX&logoColor=black"> |
|----------|:-------------:|
|__Frontend__| <img src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=React&logoColor=black"> <img src="https://img.shields.io/badge/Redux-764ABC?style=for-the-badge&logo=Redux&logoColor=black"> <img src="https://img.shields.io/badge/MUI-007FFF?style=for-the-badge&logo=MUI&logoColor=black"> <img src="https://img.shields.io/badge/Javascript-F7DF1E?style=for-the-badge&logo=Javascript&logoColor=black"> |
|__Backend__| <img src="https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=Django&logoColor=white"> <img src="https://img.shields.io/badge/Gunicorn-499848?style=for-the-badge&logo=Gunicorn&logoColor=black"> <img src="https://img.shields.io/badge/RabbitMQ-FF6600?style=for-the-badge&logo=RabbitMQ&logoColor=black"> <img src="https://img.shields.io/badge/Celery-37814A?style=for-the-badge&logo=Celery&logoColor=black"> <img src="https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=Redis&logoColor=black"> |
|__DB__| <img src="https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=MySQL&logoColor=black"> <img src="https://img.shields.io/badge/AmazonRDS-527FFF?style=for-the-badge&logo=AmazonRDS&logoColor=black"> |
|__AI__| <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=OpenCV&logoColor=black"> <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=black"> <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=black"> <img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=Google%20Colab&logoColor=black"> <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=Flask&logoColor=white"> |
|__Monitoring__| <img src="https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=Prometheus&logoColor=black"> <img src="https://img.shields.io/badge/Grafana-F46800?style=for-the-badge&logo=Grafana&logoColor=black"> |
|__Others__| <img src="https://img.shields.io/badge/Postman-FF6C37?style=for-the-badge&logo=Postman&logoColor=black"> <img src="https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=Slack&logoColor=black"> <img src="https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=Notion&logoColor=white"> <img src="https://img.shields.io/badge/GitKraken-179287?style=for-the-badge&logo=GitKraken&logoColor=black"> <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=Git&logoColor=black"> |
  
</br>



## Backend API
<img width="749" alt="스크린샷 2023-02-02 오전 2 46 07" src="https://user-images.githubusercontent.com/121562023/216704040-9b0dfc04-95ac-4dbe-bad7-f37add243f29.png">


### **users api**

- ***api/v1/users/***
    - `GET` : 아이디 중복 확인
    - `POST` : 회원가입

- ***api/v1/users/auth***
    - `POST` : 로그인

- ***api/v1/users/mypage/{case}***
  - `GET` : 마이페이지 데이터 조회 <br></br>
    > case1 : 내가 했던 이모지<br>case2 : 내가 만든 이모지

### **faces api**


- ***/api/v1/faces/tasks/***
    - `POST` : 사용자가 업로드한 사진 분석 비동기 처리

- ***/api/v1/faces/tasks/{task_id}***
    - `GET` : task id 로 분석 완료 여부 확인 및 front 에서 polling

- ***/api/v1/faces/ranking***

    - `GET` : Emoji 랭킹 조회

### **emojis api**
- ***/api/v1/emojis/***

    - `POST` : 사용자의 이모지 추가
    - `GET` : id 에 맞는 emoji 정보 조회



- ***/api/v1/emojis/pages/{page_number}***
    - `GET` : 메인 화면에 표시되는 emoji 데이터, 페이지네이션 적용

## AI
- ***OpenCV***
<img width="613" alt="Untitled" src="https://user-images.githubusercontent.com/121562023/216668440-c19afb08-a2f6-4f76-938b-de46b09b423f.png">

<img width="613" alt="AI" src="https://user-images.githubusercontent.com/121562023/216668601-c75c0bf1-1c5d-4e80-9236-3e2389bc2b64.png">

   - 얼굴과 감정 인식을 위하여 파이썬 패키지의 FER라이브러리 및 딥러닝과 OpenCV 기능을 결합하여 이미지에서 인식 기능을 수행
   - Haar Cascade classifier 또는 MTCNN(Multi-task Cascaded Convolutional Networks)을 사용하여 감정에 할당된 값의 배열 획득
   - 이를 통한 dominant한 감정 데이터를 별도로 추출, 표정에 맞는 이모지를 불러옴


- ***Rembg 라이브러리***
<img src = "https://user-images.githubusercontent.com/121562023/216721967-ab8b8826-b8a2-4b00-a7b0-e8624f3ed03f.png" width="70%">

```
image segmentation(이미지 분리) 과정을 통해 가시성을 높여준다.
```

<br></br>


## Dataset

<img width="1193" alt="스크린샷 2023-02-03 오후 8 01 02" src="https://user-images.githubusercontent.com/121562023/216587882-c3961bc4-4a23-4dcb-af5b-37839305a9aa.png">




## JWT

- used_library
    - `djangorestframework-jwt`
- JWT_Settings.py
    - set JWT’s default_settings
        
        <img width="200" alt="스크린샷 2023-02-02 오후 5 38 14" src="https://user-images.githubusercontent.com/121562023/216395399-26160577-790d-493e-93c5-18325845f6d9.png">


- Bcrypt : 데이터베이스에 유저의 정보를 저장할 때, 비밀번호와 같이 암호화가 필요한 데이터를 쉽게 다룰 수 있도록 해주는 password hashing 라이브러리이다.
- Redux : redux 를 활용하여 access-token 을 보관


## Monitoring

### Prometheus
- api 요청과 그 결과에 대한 Metric 데이터를 Prometheus 을 통해 수집
- 해당 노드의 Metric data

<img width = "1000" alt ="183143866-e2bb4650-3098-4f79-8e45-c3eb89ef032b" src="https://user-images.githubusercontent.com/121562023/216610551-9500844b-6c94-40a1-8ce1-2667cdbd454d.png">

### Grafana
- Prometheus 로 수집한 데이터를 Grafana 를 통해 시각화 한다.

<img width="1200" alt="스크린샷 2023-01-26 오후 2 41 27" src="https://user-images.githubusercontent.com/121562023/216264603-fa949b72-a62d-4f09-8a44-733572aa8aba.png">


#**Members of Team-E**
|이름|개발분야|소개페이지|
|---|---|---|
|이정우|Front-end, Back-end, Devops|https://github.com/RayLee-Kor
|박경은|Front-end, Devops|https://github.com/devGEP
|정예린|Front-end|https://github.com/Jungyell
|김대희|Back-end|https://github.com/Dan2er
|박소윤|Back-end, Devops|https://github.com/velyvelylovely
|조현진|AI|https://github.com/chojinie/
