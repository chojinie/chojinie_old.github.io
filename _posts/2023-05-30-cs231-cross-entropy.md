---
layout: post
title: cross entropy
date: 2023-05-30
description: recording_a Summary of study
tags: cs231 study AI
categories: study cs231 AI
related_posts: True
giscus_comments: true
toc:
  sidebar: left
---

## entropy(엔트로피)
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/entropy.gif" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

엔트로피는 무질서, 임의성 또는 불확실성의 상태와 가장 일반적으로 관련된 과학적 개념이자 측정 가능한 물리적 특성입니다.(wikipedia) 고등학교, 대학교에 와서까지도 나오는 용어였습니다.
이제는 정보이론 측면에서 살펴보려고 합니다. 불확실성, 무질서의 상태를 측정하기 때문에 엔트로피 값이 클 수록 데이터가 분류가 잘 되어 있지 않은 상태이고 엔트로피 값이 작을 수록 데이터가 잘 분류되어 있는 것입니다.

### 공식
엔트로피 공식은 n개의 가능한 사건으로 일반화될 수 있습니다.
어떤 사건 A가 발생할 확률을 $$ p $$ 라고 할 때 엔트로피 공식은 아래와 같습니다.
\begin{equation} h(x) = -\sum_{i=1}^n (p_i log_2(p_i)) \end{equation}
$$ log_2(p) $$에 $$ p $$를 곱한 후 모두 더한 값에 -(minus)를 취해주는 것입니다. 

이 식은 어떠한 개념에서 나오게 된 것일까요?<br>

엔트로피는 특성 사건에 대해 알기 위해 얻어야 하는 정보의 평균 비트 수입니다. 사건의 결과를 알기 위해서는 불확실성을 0으로 줄여야 합니다(즉, 확실성을 1로). 어떤 사건 A가 발생할 확률이 p라면 그 결과를 안다는 것은 불확실성을 $$ \fraction{1}{p} $$ 만큼 줄이는 것을 의미합니다. 따라서 사건 결과에 대해 알기 위해서는 $$ log\fraction{1}{p} $$ 비트 수가 필요하며 이는 $$ -log(p) $$와 같습니다. 이것은 A가 발생할 때의 엔트로피 값입니다. 마찬가지로 A가 일어나지 않을 때의 엔트로피 값은 $$ -log(1-p) $$입니다. 확률 분포가 p를 갖는 *Bernoulli* 인 경우 사건의 평균 엔트로피는 $$ -p * log(p) -(1-p) * log(1-p) $$입니다.

### 엔트로피 VS 사건의 개수
사건의 개수가 많아진다면, 엔트로피는 어떻게 될까요? 아마 높아질 것으로 직감할 수 있을 것입니다. 확률이 같은 n개의 이벤트를 선택하고 확률 분포의 엔트로피를 계산해보겠습니다.
\begin{equation} -n * \fraction{1}{n} * log\fraction{1}{n} = log(n) \end{equation}
n이 증가할 수록 log함수적으로 엔트로피는 증가하게 됩니다.

## cross entropy(교차 엔트로피, 크로스 엔트로피)

<p>KNN(K-Nearest Neighbor)분류기입니다. 우리가 타겟하는 포인트와 가까이에 있는 k개를 살펴보고 k개의 포인트가 가장 많이 속해있는 집단(class)을 정답으로 하는 분류기입니다. 
사진에서 고양이를 찾는다고 다시 생각해보겠습니다. 아래 그림처럼 어느 한 point가 우리가 찾고자하는 지점이라고 하겠습니다. k = 3으로 둘 경우, 타겟 포인트로 부터 근처에 3개의 샘플을 추출하여 어느 영역에 더 많이 속해있는지를 보고 타겟 포인트도 해당 영역의 값이라고 예측하는 것입니다.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic58.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>




## 참고
https://squarecircle.be/entropy-and-disorder-the-fate-of-all-human-enterprises/<br>
https://www.philgineer.com/2021/10/31.html<br>
https://westshine-data-analysis.tistory.com/<br>
https://www.youtube.com/watch?v=r3iRRQ2ViQM<br>
특히 수식을 이해하는 데에는 아래 사이트가 도움 되었습니다.
**https://towardsdatascience.com/entropy-cross-entropy-and-kl-divergence-17138ffab87b** <br>