---
layout: post
title: support \vector machine(SVM)
date: 2023-06-01
description: recording_a Summary of math
tags: study math
categories: study math
related_posts: True
giscus_comments: true
toc:
  sidebar: left
---

처음 접하게 된 SVM을 정확히 이해하려고 노력했습니다. UNIST 유재준 교수님의 블로그(하단의 참고에 출처)와 Mathmatic for ML교재를 바탕으로 공부 내용을 정리하고자 합니다. SVM은 매우 아름답고 탄탄한 이론적인 배경을 바탕(Steinwart and Christmann, 2008)으로 정교하게 고안된 기계학습 알고리즘이며, 실제 적용이 여러 모로 쉽고 성능이 강력하며 따라서 실전적이라는 점이 매력적이라고 합니다.

## Classification with Support \Vector Machines

많은 경우에 우리는 다양한 선택지 중에서 옳은 답을 결정하는 머신러닝 알고리즘을 바랄 것입니다. 예를 들어, 스팸 메일 분류 기능에서는 스팸 메일과 정상 메일을 분류해 낼 수 있죠. 이렇게 오직 두개의 결과만을 바라 보고 어떤 결과에 해당되는지를 분류해내는 분야를 Binary Classification(이진 분류)이라 합니다. 출력이 분류값으로 가질 수 있는 오직 두개의 클래스를 \{+1, -1 \} 으로 나타내겠습니다. 아래와 같은 수식으로 표현하게 됩니다.

\begin{equation} f : \mathbb{R}^D \rightarrow \{ +1, -1 \} \end{equation}

각각의 example (데이터 포인트) $$ x_n $$ 은 D개의 실수로 구성된 피처 벡터로 나타냅니다. 레이블은 일반적으로 양성 클래스(+1)와 음성 클래스(-1)로 구분됩니다. 이진 분류작업에 사용되는 SVM (Support \Vector Machine)이라고 하는 접근 방식을 소개하겠습니다. 

SVM이 풀고자하는 문제는 다음과 같습니다.

```
"How do we divide the space with decision boundaries?"
```

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/svm/pic4.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

위 그림에서 보면 '+' example과 '-' example을 어떻게 구별 할 수 있을까?
어떻게 최대한 그 둘 간의 구분을 지어 놓을까에 대해 고민하는 것으로 해석 할 수 있겠습니다.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/svm/pic5.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

이런식으로 -와 + example이 가장 맞닿아 있는 부분들(노란 실선)로 부터 특정 간격만큼의 거리가 벌려져서 가장 넓어 질 수 있는 라인(빨간 점선)을 정하면 될 것 같습니다.

<details>
<summary>접기/펼치기</summary>

살짝 복잡하게 말해보면, 이는 회귀와 마찬가지로 binary 레이블 $$ y_n \in \{+1, -1\} $$ 와 짝을 이루는 example $$ x_n \in \mathbb{r}^D $$ 의 집합에서 지도 학습 task를 갖고 있습니다. example-레이블 쌍 {(x1, y1), ..., (xN, yN)}로 구성된 훈련 데이터 세트가 주어졌을 때, 최소의 분류 오류를 얻는 모델의 매개변수를 추정하는 것 입니다. 선형/비선형 모델을 모두 고려해야하지만, 당장은 선형 모델만을 고려하겠습니다.

이진 분류를 SVM을 사용하여 설명하는 데에는 두 가지 주요 이유가 있습니다. 첫째, SVM은 지도 학습의 기하학적인 관점을 고려할 수 있게 해줍니다. 두 번째는 SVM의 최적화 문제가 해석적인 해를 가지지 않아 다양한 최적화 도구를 활용해야 한다는 점입니다.

SVM의 기계 학습 관점은 최대 우도 관점과 약간 다릅니다. 최대 우도 관점은 데이터 분포의 확률적인 관점을 기반으로 모델을 제안하고, 이를 기반으로 최적화 문제를 도출합니다. 반면, SVM 관점은 기하학적 직관에 기반하여 훈련 중에 최적화되어야 하는 특정한 함수를 설계하는 것으로 시작합니다. SVM의 경우, 훈련 데이터에서 최소화되어야 하는 손실 함수를 설계하기 시작합니다. 이는 경험적 위험 최소화 원칙을 따릅니다.

</details>

### Decision Rule

점선(decision boundary)를 정하기 위해 decision rule에 대한 설정이 선행되어야 합니다. example과 점선 간의 관계를 나타내기 가장 쉬운 방법은 "거리", "방향"과 관련된 식을 도출해내는 것입니다. 그렇기 때문에 $$ \vec{w} $$ 를 하나 그려보겠습니다. 간격의 중심선(빨간 점선)에 대해 직교하는 벡터입니다.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/svm/pic6.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

그리고 임의의 example $$ \vec{u} $$ 가 있을 때 간격을 기준으로 오른쪽에 속할지 왼쪽에 속할지를 알아내야 합니다. 여기서 내적의 개념을 생각해 볼 수 있습니다. 내적은 하나의 벡터가 다른 벡터로 projection하는 것으로 이해할 수 있죠.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/svm/pic7.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

$$ \theta $$ 만큼의 각도를 가지며 $$ \vec{a} $$와 $$ \vec{b} $$가 내적하는 경우를 상상해봅시다. 만약 두 벡터가 X,Y평면에 있다고 생각하면 $$ \vec{a} $$는 x, y 성분으로 분리할 수 있겠죠. $$ \vec{b} $$ 가 x축이라고 여기면, projection된 a는 b벡터 방향(x축 방향)의 x성분 만큼의 크기를 갖는 $$ acos\theta $$의 벡터 형태로 변하게 됩니다. 이를 b와 곱한 것이 내적의 표현입니다.

다시 $$ \vec{w} $$, $$ \vec {u} $$의 입장으로 돌아가 보겠습니다. $$ \vec {u} $$는 $$ \vec {w} $$와 $$ \theta $$ 만큼 각도를 가지며 u는 w방향 성분만큼 projection되어 $$ ucos\theta $$ 벡터로 변할 것입니다. 그리고 $$ \vec{u} \cdot \vec{v} = \vec{u}\vec{v}cos\theta $$ 의 크기를 갖는 선이 될 것입니다. 그 선의 크기(길이)의 의미는 그래프의 원점으로 부터 간격이 있는 방향으로 얼마만큼 멀리 떨어져 있느냐를 얘기합니다.

길이가 너무 길어서 간격을 벗어나면 '+', 너무 짧아서 간격에 못미치면 '-'를 분류해 내는 것입니다. 임의의 상수 b와 크기를 비교하는 식으로 일반화 시킬 수 있습니다.
\begin{equation}
\vec{w} \cdot \vec{u} + b \ge 0 \qquad then \; '+'
\end{equation}

하지만 일반화 식에서 \vec{w} 와 b는 어떤 값을 잡아야하는지 전혀 알 수 가 없습니다. under constraint 상황이기 때문에 constraint (제약) 조건을 추가하는 작업을 아래에서 해 나갈 예정입니다.

### Design and Add additional constraints

위 식을 각 클래스 별로 식으로 표현해보겠습니다.

\begin{equation} \vec{w}\{x_+} + b \ge 1
\end{equation}

\begin{equation} \vec{w}\{x_-} + b \le -1
\end{equation}

즉 decision rule이 최소한 1보다 크거나 -1보다 작은 값을 주도록 해봤습니다. 두 개의 식을 하나의 변수를 추가하여 하나의 식으로 변경해 보도록 하죠.(변수를 추가하는 수학적 의미는 거창한게 아니라 편하려고 도입한 것입니다.)

$$ y_i = \Big\{ 1 for '+' \\ -1 for '-' $$

$$ y_i $$ 를 (3) 식에 각각 곱해 줍니다.
아래와 같이 두개의 식을 하나의 식으로 묶어서 표현할 수 있게 됩니다.
\begin{equation}

y_i ( \vec{w}\{x_+} + b ) \ge 1

\end{equation}

1을 좌변으로 옮기면 다음과 같이 표현할 수 있게 됩니다.

\begin{equation}
y_i ( \vec{w}\{x_+} + b ) -1 \ge 0
\end{equation}


SVM을 훈련하기 위한 최적화 문제를 유도해 보도록 하겠습니다. 직관적으로, 우리는 이진 분류 데이터를 상상해볼 수 있습니다. 데이터는 위의 그림처럼 Hyperplane(초평면)으로 분리될 수 있습니다. 여기에서 모든 example $$ x_n $$ (2차원 벡터)은 2차원 위치 $$ (x_n^{(1)}, x_n^{(2)}) $$이고, 해당하는 이진 레이블 $$ y_n $$은 두 가지 다른 기호 (주황색 엑스표시 또는 파란색 원모양) 중 하나입니다. "Hyperplane"은 기계 학습에서 흔히 사용되는 용어입니다. Hyperplane은 차원이 D - 1인 (해당하는 벡터 공간이 차원 D인 경우) 아핀 sub space입니다. 이 예제들은 두 개의 클래스로 구성되어 있으며 (두 가지 가능한 레이블이 있음), 이들을 직선으로 그려서 분리/분류할 수 있도록 특징들 (example을 나타내는 벡터의 구성 요소들)이 배열되어 있습니다.

두 클래스를 나누는 선형 separator를 찾기 위한 아이디어를 공식화 해보려고합니다. margin(마진)의 개념을 소개한 다음, 분류 에러를 야기하여 example을 "wrong" 부분에 놓이게 하도록 선형 separator의 개념을 확대하겠습니다. 두 가지 관점에서의 동등한 SVM을 공식화하는 방식을 얘기할 것입니다. : 첫 번째는 geometric 관점이며, 두 번째는 loss function 관점입니다. Lagrange multipliers를 이용하여 SVM의 dual version을 유도할 것입니다. dual SVM은 SVM 공식화의 세 번째 관점을 제공합니다.(각 클래스의 example의 convex hulls로 표현하는 관점) 이 후에는 커널(kernels)에 대해 간략히 설명한 후, 비선형 커널 SVM 최적화 문제를 수치적으로 어떻게 해결하는 지에 대한 설명으로 마무리 하겠습니다.

## Separating Hyperplanes

두 개의 벡터 $$ x_i, x_j $$가 example들로 주어졌다고 하겠습니다. 이 두 벡터 간의 유사성을 계산하는 한 가지 방법은 내적을 하는 것입니다 $$ <x_i, x_j> $$. 내적은 두 벡터 사이의 각도와 큰 관련이 있습니다. 내적 값은 각 벡터의 길이(norm)에 따라 달라집니다. 또한, 내적은 수학적으로 직교성(orthogonality)과 투영(projections)과 같은 기하학적 개념을 엄격하게 정의할 수 있게 해줍니다. 많은 분류 알고리즘의 주요 아이디어는 데이터를 $$ \mathbb{R}^D $$ 공간에 표현한 다음, 이 공간을 분할하는 것입니다. 이상적으로는 동일한 레이블을 가진 example들이 같은 분할에 속하고 다른 것들은 속하지 않도록 분할하는 것입니다. binary classification에서는 양성/음성 클래스 두 부분으로 "hypterplane"을 이용하여 공간을 분할합니다. 모든 example은 $$ \mathbb{R}^D $$ data space의 element(요소)라고 둡시다. $$ w \in \mathbb{R}^D $$ 그리고 $$ b \in \mathbb{R} $$ 에 의해 매개변수화된 함수 f를 다음과 같이 표현할 수 있습니다.

\begein{equation}
f: \mathbb{R}^D \rightarrow \mathbb{R} \\
x \mapsto f(x) := <w,x> + b \\
\end{equation}

hyperplane은 아핀 subspace입니다. 따라서 이진 분류 문제에서 두 클래스를 분리하는 초공간은 다음과 같이 정의됩니다.

\begein{equation}
\{x \in \mathbb{R}^D : f(x) = 0 \}
\end{equation}

아래 그림처럼 초공간벡터 w는 하이퍼플레인에 수직인 벡터이고 b는 절편(intercept)입니다.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/svm/pic2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

위 식에서 $$ w $$ 가 초공간에 수직인 normal \vector임을 증명하기 위해 초공간 상의 임의의 두 example인 $$ x_a와 x_b $$를 선택하고 그 사이의 벡터가 $$ w $$와 수직임을 보일 수 있어야 합니다. 이를 수식으로 나타내면 아래와 같습니다. (두 번째 수식은 내적의 선형성에 의해 얻어집니다.)

\begein{equation}
f(x_a) - f(x_b) = <w, x_a> + b - (<w, x_b> + b)
= <w, x_a - x_b>\end{equation}

초공간 상의 점 들인 $$ x_a, x_b $$를 골랐으므로 $$ f(x_a), f(x_b) = 0  $$ 입니다. 나아가 $$ <w, x_a - x_b> = 0 $$d입니다. 두 벡터가 orthogonal할 때 그들의 내적은 0이 나온다는 개념을 다시 떠올려봅시다. 그렇다면, 초공간 상의 모든 벡터에 orthogonal한 $$ w $$를 얻을 수 있게 됩니다. 벡터의 개념은 다양하게 쓰일 수 있씁니다. 다만, 지금 설명하는 매개 변수 벡터 $$ w $$는 방향을 나타내는 기하학적 벡터로 생각합니다. 반면, example로 표시되는 $$ x $$의 벡터는 데이터 포인트로 나타냅니다. $$ x $$를 basis에 대한 좌표를 갖는 벡터로 간주하는 것입니다. 테스트 example이 주어지면, 어느 쪽에 위치하는지에 따라 양성 또는 음성으로 분류합니다. 이는 example이 하이퍼플레인의 어느 쪽에 있는지를 기준으로 판단합니다. hyperplane을 나타내는 식은 방향까지 정해줍니다. 즉, 초공간의 양/음 방면을 결정하는 것입니다. 그래서 $$ f(x_test) \ge 0 $$ 라면 + 1 로 구할 수 있고 반대는 -1로 구하게 되는 것입니다. 한 번 그림을 상상해보죠. 양의 값을 갖는 example은 hyperplane의 "위"에 위치해 있을 것이고, 음의 값을 갖는 것은 "아래" 위치해 있을 것입니다. 


## 참고
MATHEMATICS FOR MACHINE LEARNING(1st Edition) by Marc Peter Deisenroth
https://jaejunyoo.blogspot.com/search?q=svm