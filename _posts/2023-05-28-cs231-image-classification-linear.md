---
layout: post
title: cs231n Summary - Image Classification with Linear Classifiers
date: 2023-05-28
description: recording_a Summary of lecture
tags: cs231 study AI
categories: study cs231 AI
related_posts: True
giscus_comments: true
toc:
  sidebar: left
---

## Image Classification
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic44.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic45.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>    
</div>

컴퓨터가 고양이 사진을 본다면 밝기 성분을 [0, 255] 범위의 정수의 집합으로 표현할 것입니다. 예시의 고양이 사진의 전체사이즈는 대략 가로 세로 1:2 비율이라고 본다면, 200px X 400px크기의 이미지로 둘 수 있습니다. 또한 color이미지이므로 RGB 3channels로 구성되어 있습니다. 네모난 창문(window)을 보면 왼쪽 상단에서 우측 상단으로 탐색하면서 해당 위치는 RGB가 각각 얼마만큼 (0~255 사이) 성분이 있는지를 나타내는 값으로 저장됩니다.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/rgb.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

하지만, 만약 고양이 사진을 다른 각도에서 찍거나 조명이 달라지거나 한다면 밝기 성분은 당연히 달라질 수 밖에 없겠죠. 사람은 같은 고양이로 인식할 것입니다. 누워있던 숨어있던 뒤돌아 있던, 하지만 컴퓨터는 어렵죠.<br>

Challenges of recognition : 
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic49.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

물론 고전적인 Computer vision에서는 edge detection 등 많은 방식으로 Classification task(분류 문제)를 풀려고 했습니다. 하지만 이렇다할 성능을 보이지는 못했고, 현대에 와서는 Data-driven 방식의 머신러닝 기법을 적용하게 되었습니다.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic53.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic54.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

### Nearest Neighbor Classifier

<p>KNN(K-Nearest Neighbor)분류기입니다. 우리가 타겟하는 포인트와 가까이에 있는 k개를 살펴보고 k개의 포인트가 가장 많이 속해있는 집단(class)을 정답으로 하는 분류기입니다. 
사진에서 고양이를 찾는다고 다시 생각해보겠습니다. 아래 그림처럼 어느 한 point가 우리가 찾고자하는 지점이라고 하겠습니다. k = 3으로 둘 경우, 타겟 포인트로 부터 근처에 3개의 샘플을 추출하여 어느 영역에 더 많이 속해있는지를 보고 타겟 포인트도 해당 영역의 값이라고 예측하는 것입니다.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic58.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


</p>
task 수행을 위해 모델이 필요하다고 했었죠. 하지만 KNN의 경우 특별한 모델이 필요한 것은 아니고 training data(학습 데이터)의 정답을 모두 저장해두었다가 나중에 test data(테스트 데이터)를 모델에 넣었을 때 정답과 가장 유사한 결과를 예측해내어 분류 task를 수행하는 것이 전부입니다. 그렇기 때문에 복잡한 인공지능 여타 모델과 달리 lazy model이라고 이야기 하기도 합니다.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic55.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

고양이라는 (Label)을 달고 있는 training data를 저장해두었다가, 나중에 query로 들어오는 Test data와의 거리 비교(Distance Metric)를 하여 그 차이가 가장 작은 데이터를 고양이라고 분류하는 거죠.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic56.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


Distance Metric 즉, 거리 차이를 비교하는 방법은 대표적으로 두가지가 있습니다. L1 distance와 L2 distance 입니다.

L1 distance: \begin{equation} d_1(I_1, I_2) = \sum\limits_{p}|I_1^{p} - I_2^{p}| \end{equation}
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic57.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

L2 distance: \begin{equation} d_1(I_1, I_2) = \sqrt{\sum\limits_{p}(I_1^{p} - I_2^{p})^2} \end{equation}

#### Hyperparameter setting
<p>그렇다면, 얼마만큼의 k를 줘야 최적의 답을 끌어내는 모델일까요? 혹은 어떠한 distance metric을 사용해야 최고의 모델이 될까요?<br>
이러한 요소들이 우리들이 학습하면서 수정해 나아가야 할 KNN모델의 Hyperparameter라고 합니다.
매우 어려운 부분이고, 데이터셋에 의존적입니다. 모든 알고리즘을 실행해보고 어떤 것이 가장 잘 동작하는지 확인을 해야합니다.
</p>
<p>
아래 이미지 두 장이 설명이 잘 되어 있습니다. 즉, dataset은 Training, Validation, Test의 3분할이 가장 효과적일 수 있다는 것이죠. 나아가 Cross-Validation을 통해 Fold부분을 순환하여 validation data로 중간평가를 하게 된다면 Small dataset에 한해서는 꽤 효과적일 수도 있다고 합니다. 단 딥러닝 분야에서는 잘 사용되지 않는다고 강의에서는 얘기하고 있습니다.
</p>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic59.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic60.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic61.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

#### k-Nearest Neighbor with pixel distance never used

k-Nearest Neighbor 알고리즘에서 픽셀 간의 거리를 사용하지 않는 이유는 다음과 같습니다:

1. 차원의 저주 (Curse of Dimensionality): 이미지 데이터의 경우 각 픽셀은 하나의 차원으로 간주됩니다. 이미지의 크기가 커질수록 차원의 수가 급격히 증가하게 되는데, 이는 차원의 저주로 알려진 문제를 야기할 수 있습니다. 차원의 저주는 고차원 공간에서 데이터 간의 거리 측정이 더 어렵고, 더 많은 데이터가 필요하게 만들어서 k-최근접 이웃 알고리즘의 성능을 저하시킵니다.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic62.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
2. 픽셀 간의 거리는 이미지의 의미를 적절하게 반영하지 않을 수 있습니다. 이미지는 일반적으로 시각적인 패턴, 구조, 텍스처 등을 포함하고 있으며, 이러한 특징들은 단순히 픽셀 값의 거리로만 표현하기에는 제한적일 수 있습니다. 따라서 픽셀 간의 거리를 사용하지 않는 대안적인 특징 추출 방법이나 거리 측정 방법을 사용하는 것이 더 효과적일 수 있습니다.

3. 픽셀 간의 거리는 이미지의 변형에 민감할 수 있습니다. 이미지에 대한 작은 변형이나 노이즈의 증가가 픽셀 값의 큰 변화로 나타날 수 있기 때문에, 픽셀 간의 거리를 직접 사용하면 이러한 변형에 매우 민감한 모델이 만들어질 수 있습니다. 이는 모델의 일반화 성능을 저하시킬 수 있습니다.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic63.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

따라서, 이미지 데이터의 경우에는 픽셀 간의 거리를 직접 사용하지 않고, 보다 효과적인 특징 추출 방법과 거리 측정 방법을 활용하는 것이 일반적입니다.


## Linear Classifier
선형 분류기(Linear Classifier)는 입력 데이터를 선형 경계로 구분하는 분류 알고리즘입니다. 주어진 입력 데이터에 대해 각각의 특성을 가중치와 결합하여 선형 함수를 생성하고, 이 함수의 결과를 기반으로 데이터를 클래스로 분류합니다.

선형 분류기는 입력 데이터를 공간상의 점들로 표현하고, 클래스를 나누는 하이퍼플레인(hyperplane)이라는 선형 경계로 구분합니다. 이 때, 하이퍼플레인은 입력 특성의 가중치와 절편(bias)으로 정의됩니다. 예를 들어, 이진 분류에서는 하이퍼플레인이 입력 공간을 두 영역으로 나누는 직선이 될 수 있습니다.

선형 분류기는 간단하고 해석하기 쉬우며, 많은 문제에 효과적으로 적용될 수 있습니다. 그러나 입력 데이터가 비선형적인 관계를 가지거나, 클래스 간의 결정 경계가 비선형적일 경우에는 선형 분류기의 성능이 제한될 수 있습니다. 이러한 경우에는 비선형 분류기를 사용하거나, 선형 분류기를 변형하여 비선형성을 고려할 수 있는 방법들을 적용할 수 있습니다.

이 기법에는 원본 데이터를 클래스 점수로 바꿔 주는 점수 함수(score function)와, 예측된 점수들과 참값 레이블이 얼마나 일치하는지를 수량화한 손실 함수(loss function), 이렇게 두 가지 중요한 요소가 있습니다. 이들을 이용하면 이미지 분류 문제를 점수 함수의 파라미터에 대해 손실 함수를 최소화하는 최적화 문제(optimization problem)로 바꿀 수 있습니다.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic64.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

위 그림 예시를 보면, x라는 이미지가 들어갔을 때, W라는 가중치와 함께 어떠한 함수f를 구하면, Wx + b와 같은 형태가 나오게 됩니다. 차원 값을 보게 된다면, 10개의 숫자가 10개의 클래스에 할당되는 점수값으로 반환되는 것을 알 수 있습니다.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic65.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

맨 마지막 layer에서 linearity를 주어, 최종적으로 찾고자하는 클래스에 대응되는 값을 구합니다.

이미지를 3클래스로 분류하는 것을 예시로 들면 아래와 같이, 대수적/기하학적/시각적 관점에서 해석할 수 있습니다.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic66.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

선형 분류기는 이미지의 전체 3개 채널의 모든 픽셀값들의 가중합(weighted sum)을 구해 각 클래스의 점수를 구합니다. 가중치에 어떤 값이 설정되었는지에 따라 점수 함수는 이미지의 특정 위치의 특정 색을 좋아할 수도(가중치 부호가 +일 때) 좋아하지 않을 수도(가중치 부호가 -일 때) 있습니다. 예를 들어, "배(ship)" 클래스로 분류될 이미지에는 이미지 한편에 파란색이 많을 것입니다(아마도 물을 나타낼 겁니다). 그렇다면 "배" 분류기는 파란색 채널의 가중치 값에는 양수가 많을 것이고(= 파란색의 존재는 "배" 클래스의 점수를 높입니다.), 빨간색/초록색 채널의 가중치 값에는 음수가 많을 것입니다(= 빨간색/초록색의 존재는 "배" 클래스의 점수를 낮춥니다). 이미지에서는 좋지 않은 가중치로 인해 오답이 발생했습니다.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic67.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
가중치에 대한 또 다른 해석은 가중치의 각 행이 각각 클래스 하나에 대한 템플릿(template) 또는 프로토타입(prototype)이라 보는 것입니다. 클래스 점수는 클래스의 템플릿과 이미지를 하나씩 내적(inner product, dot product)하여 구합니다. 선형 분류기는 학습된 템플릿들을 이용해 가장 "잘 맞는" 템플릿을 찾는 것이라 할 수 있습니다. 즉 선형 분류기는 여전히 최근접 이웃(Nearest Neighbor)을 찾고 있는데, 이번에는 수천장의 모든 학습 이미지들을 다 가지고 있는 것이 아니라 각 클래스 당 딱 한장의 템플릿 이미지만 가지고, L1, L2 Distance를 사용하는 대신 음의 내적(negative inner product)을 사용하는 것입니다. 선형 분류기가 가지고 있는 템플릿 이미지는 학습을 통해 만들어진다. 템플릿 이미지는 학습 데이터의 사진 중 한 장일 필요는 없습니다.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic68.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
이미지 공간을 표현한 이미지. 데이터 셋의 각 이미지들은 하나의 점으로, 선형 분류기는 선으로 표현되어 있습니다. 예를 들어 자동차 분류기(빨간색)의 경우 공간 상에서 자동차 클래스 점수가 0이 나오는 모든 점들을 이은 것입니다. 이때 빨간 화살표는 클래스 점수가 증가하는 방향을 보여줍니다. 즉, 빨간 선 오른편의 모든 점들은 (화살표 방향으로 선형적으로 증가하는) 양수 점수를 가지고, 빨간 선 왼편의 모든 점들은 (화살표 방향으로 선형적으로 감소하는) 음수 점수를 가집니다.

### 좋은 가중치(Weight) 고르기

<p>
손실함수를 통해 훈련 데이터의 점수에 대한 불만을 측정하는 손실 함수를 정의하겠습니다. 그리고 손실 함수를 최소화하는 매개변수를 효율적으로 찾는 방법을 고안해보겠습니다. (최적화)

SVM(Multiclass Support Vector Machine) 손실 함수를 살펴보겠습니다. 지도학습(Supervised Learning)에서 사용되는 분류(Classification)와 회귀(Regression) 문제를 해결하기 위한 알고리즘입니다. SVM은 데이터를 고차원 공간으로 매핑하여 최적의 경계를 찾는 방식으로 작동합니다.

SVM의 주요 아이디어는 데이터를 분류할 수 있는 최적의 결정 경계를 찾는 것입니다. 이 결정 경계는 클래스 간의 간격(마진)을 최대화하고, 데이터 포인트와 가장 가까운 일부 데이터 포인트인 서포트 벡터(Support Vector)와의 거리를 최소화하려고 합니다. SVM은 선형 분류에서는 선형 경계를 찾는 선형 SVM과, 비선형 분류에서는 커널 트릭을 사용하여 비선형 결정 경계를 찾는 비선형 SVM으로 구분됩니다.

SVM은 다양한 장점을 가지고 있습니다. 예를 들어, SVM은 고차원 데이터에서도 잘 작동하며, 이상치(outliers)에 대해 강건한 특성을 가집니다. 또한, 커널 트릭을 사용하여 비선형 문제를 해결할 수 있으며, 일반화(generalization) 성능이 좋다고 알려져 있습니다.

SVM은 이진 분류뿐만 아니라 다중 클래스 분류에도 사용될 수 있으며, 회귀 문제에도 적용할 수 있습니다. SVM은 다양한 실제 응용 분야에서 폭넓게 사용되고 있으며, 머신러닝에서 중요한 알고리즘 중 하나입니다.
</p>
<p>
수식으로 표현해보겠습니다. $$ i $$ 번째 이미지의 전체 픽셀을 하나의 열 벡터로 만든 $$ x_i $$ 와 해당 미지의 lsbel $$ y_i $$ 가 있을 경우, score function(f)는 $$ x_i $$를 입력받아 클래스 점수를 나타내는 벡터 $$ s=f(x_i, W) $$를 출력합니다.(s는 점수 score를 의미) $$ x_i $$ 의 $$ j $$ 번째 클래스의 점수는 $$ s_j $$의 $$ j $$번째 요소인 $$ s_j = f(x_i, W)_{j} $$라 표현할 수 있다.
</p>

$$ i $$번째 이미지에 대한 SVM 손실 함수는 다음과 같이 일반화할 수 있다.

\begin{equation} L_i = \sum\limits_{j\neq y_i}max(0, s_j - s_{y_i} + \mathit{\Delta} \end{equation} 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic69.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic70.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic71.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic72.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
