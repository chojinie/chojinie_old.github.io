---
layout: post
title: cs231n Summary - Regularization
date: 2023-06-03
description: recording_a Summary of lecture
tags: cs231 study AI
categories: study cs231 AI
related_posts: True
giscus_comments: true
toc:
  sidebar: left
---

## 배경
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic77.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

손실 함수에는 한가지 오류가 있습니다. 이를 위해 Parameter set "W"를 갖는 데이터셋이 있다고 가정해봅시다. W는 모든 example에 대해서 정확히 분류해냅니다. 모든 스코어 점수는 모든 마진을 충족하기 대문에 $$ L_i = 0 \; \text{for all} \; i $$ 라고 둘 수 있습니다. 여기서 발생하는 문제는 W의 set이 반드시 하나만이라고 볼 수 는 없다는 점입니다.

간단히 예시를 들어 이해해 보겠습니다. 매개 변수 W가 모든 example을 올바르게 분류하여 example에 대한 loss가 0이 나온다면, parameter의 임의의 상수가 곱해진 $$ \lambda W (\lambda > 1) $$ 도 loss가 0이 됩니다. 왜냐하면 이 변환은 모든 점수의 크기를 균등하게 늘리기 때문에 그들의 절대적인 차이도 균등하게 늘어납니다. 예를 들어, 올바른 클래스와 가장 가까운 잘못된 클래스 사이의 점수 차이가 15였다면, W의 모든 요소를 2배로 곱하면 새로운 차이는 30이 됩니다. 이러한 모호성을 없애기 위해 특정 가중치 세트 W에 대한 선호도(preference)를 넣어줘야 합니다.

## 방법론

손실함수에 regularization penalty를 줌으로써 구현할 수 있습니다. 가장 보편적인 regularization penaly는 L2 norm 형태입니다. 가중치의 제곱을 더해서 큰 가중치에 패널티를 주는 방식으로 작동합니다. L2 norm을 최소화하는 것은 가중치 값을 작게 유지하는 것을 의미합니다. 따라서 큰 가중치를 줄이고 작은 가중치를 선호하는 경향이 생기게 됩니다. 이는 모델이 더 일반화되고 오버피팅을 피할 수 있도록 도와줍니다.

\begin{equation} \mathbf{R(W)} = \sum_{k}\sum_{l}(W_{k,l})^2 \end{equation}

위 식에서 W의 모든 요소를 제곱하여 더합니다. 정규화 함수는 데이터에 의존하는 것이 아니라 가중치에만 기반하므로 데이터와는 무관합니다. 정규화 페널티를 포함하여 전체 다중 클래스 SVM loss를 완성할 수 잇으며, 이는 데이터 손실과 정규화 손실로 구성됩니다. 즉, SVM은 다음과 같이 표현할 수 있습니다.

\begin{equation} L = \frac{1}{N}\sum_i{L_i} + \lambda R(W) \end{equation}

## L2 regularization의 장점

L2 norm을 사용하면 SVM에서 최대 마진 속성으로 이어지게 됩니다. 또 다른 이점은 큰 가중치에 페널티를 줘서 모델의 일반화를 개선한다는 것입니다. 이는 어떤 입력 차원도 그 자체로 점수에 매우 큰 영향을 미칠 수 없음을 의미하기 때문입니다. 예를 들어 입력 벡터 $$ \vec{x} = [1,1,1,1] $$ 이고, 두개의 가중치 벡터 $$ \vec{w_1} = [1,0,0,0] $$ , $$ \vec{w_2} = [0.25,0.25,0.25,0.25] $$ 가 있다고 가정하겠습니다. 이 경우 $$ \vec{w_1}^T{\vec{x}} = \vec{w_2}^T{\vec{x}} = 1 $$로 동일한 것을 알 수 있습니다. 하지만 $$ \vec{(w_1)} $$의 L2 penalty는 1.0인 반면 $$ \vec{w_2}^T{\vec{x}} $$ 의 것은 0.5일 뿐입니다.( $$ 0.25^2 + 0.25^2 + 0.25^2 + 0.25^2 = 0.25, $$ L2 Norm을 계산하게 되면 최종적으로 루트를 씌우기 때문에) 그래서 L2 패널티에 따르면, 가중치 벡터 w2가 더 낮은 정규화 손실을 가지기 때문에 선호됩니다. 직관적으로 이해하면, w2의 가중치는 더 작고 퍼져있기 때문입니다.



## 참고
http://cs231n.stanford.edu/schedule.html
