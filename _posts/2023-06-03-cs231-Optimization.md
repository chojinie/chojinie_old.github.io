---
layout: post
title: cs231n Summary - Optimization
date: 2023-06-03
description: recording_a Summary of lecture
tags: cs231 study AI
categories: study cs231 AI
related_posts: True
giscus_comments: true
toc:
  sidebar: left
---

## Optimization
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic78.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

최적화는 주어진 모델과 손실 함수에 대해 가장 적절한 Parameter 값을 찾아내는 것으로, 학습 알고리즘을 사용하여 이를 수행합니다. 최적화는 모델을 훈련시키고 예측 성능을 개선하기 위해 필수적인 단계입니다.

## 전략

### 첫 번째 전략 : Random Search
최적화를 하기 위해 생각을 한 번 전개해보도록 하겠습니다. 손실슬 최소화 시키는 parameter(가중치 등)를 구하기 위해서 가장 간단한 아이디어는 단순하게도 최대한 많은 가중치를 모델에 대입해서 얼마나 정확도를 보이는지(손실 함수 작은지) 테스트해보는 것입니다.

```
# assume X_train is the data where each column is an example (e.g. 3073 x 50,000)
# assume Y_train are the labels (e.g. 1D array of 50,000)
# assume the function L evaluates the loss function

bestloss = float("inf") # Python assigns the highest possible float value
for num in range(1000):
  W = np.random.randn(10, 3073) * 0.0001 # generate random parameters
  loss = L(X_train, Y_train, W) # get the loss over the entire training set
  if loss < bestloss: # keep track of the best solution
    bestloss = loss
    bestW = W
  print 'in attempt %d the loss was %f, best %f' % (num, loss, bestloss)

# prints:
# in attempt 0 the loss was 9.401632, best 9.401632
# in attempt 1 the loss was 8.959668, best 8.959668
# in attempt 2 the loss was 9.044034, best 8.959668
# in attempt 3 the loss was 9.278948, best 8.959668
# in attempt 4 the loss was 8.857370, best 8.857370
# in attempt 5 the loss was 8.943151, best 8.857370
# in attempt 6 the loss was 8.605604, best 8.605604
# ... (trunctated: continues for 1000 lines)
```
```
# Assume X_test is [3073 x 10000], Y_test [10000 x 1]
scores = Wbest.dot(Xte_cols) # 10 x 10000, the class scores for all test examples
# find the index with max score in each column (the predicted class)
Yte_predict = np.argmax(scores, axis = 0)
# and calculate accuracy (fraction of predictions that are correct)
np.mean(Yte_predict == Yte)
# returns 0.1555
```

이를 구현한 코드가 진행되면서 성능이 좋아지는 것을 볼 수 있습니다. 이를 통해 첫 번째 전략의 핵심 아이디어는 '반복'과 '개선'임을 알 수 있습니다. 임의의 W로 모델 테스트를 시작한 다음 반복하여 손실을 줄이는 것입니다. 하지만, 우리가 리신이 되어 등산로 입구로 내려가는 등산객이라고 한 번 생각해봅시다. CIFAR-10의 예에서 산에 있는 언덕의 개수가 W의 차원인 10 X 3072개 만큼 있을 겁니다... 어렵겠죠? 그만큼 이 방법은 좋지 않다는 것을 바로 느낄 수 있을 겁니다.

### 두 번째 전략 : Random Local Search

생각할 수 있는 첫 번째 전략은 임의의 방향으로 한 발을 뻗은 다음 내리막으로 이어지는 경우에만 발을 내딛는 것입니다.
구체적으로, 무작위로 선택한 초기 W로 시작하고, 이에 대해 무작위로 작은 움직임인 δW를 생성합니다. 그리고 만약 변동을 가한 W+δW에서의 손실이 더 낮다면, 우리는 더 좋은 가중치를 찾은 것으로 업데이트를 수행합니다. 즉 손실을 줄이는 방향으로 학습하는 것이죠. 
```
W = np.random.randn(10, 3073) * 0.001 # generate random starting W
bestloss = float("inf")
for i in range(1000):
  step_size = 0.0001
  Wtry = W + np.random.randn(10, 3073) * step_size
  loss = L(Xtr_cols, Ytr, Wtry)
  if loss < bestloss:
    W = Wtry
    bestloss = loss
  print 'iter %d loss is %f' % (i, bestloss)
```
하지만, 이것도 실제로 구현해 보면, 분류 정확도 21.4% 정도 밖에 얻지 못합니다. 또한 여전히 컴퓨팅 파워를 쓸데 없이 많이 잡아 먹습니다.

### 세 번째 전략 : Following the Gradient

좋은 방향을 무작위로 탐색할 필요는 없습니다. 손실을 가장 급격하게 감소시키는 방향을 수학적으로 계산하여 가중치 벡터를 변경해야 할 최적의 방향을 구할 수 있습니다. 이 방향은 손실 함수의 그래디언트와 관련이 있습니다. 두 번째 전략과 컨셉이 비슷하죠. 그래디언트는 입력 공간의 각 차원에 대한 기울기의 벡터(도함수:derivatives)입니다. 수학적 표현으로는 아래와 같이 나타낼 수 있습니다.

\begein{equation}
\frac{df(x)}{dx} = \lim{h \to 0}\frac{f(x+h)-f(x)}{h}
\end{equation}


**Gradient 계산방법** <br>
그래디언트를 계산하는 방법에는 두 가지가 있습니다. 느리고 대략적이지만 쉬운 방법(Numerical 
그래디언트)과 빠르고 정확하지만 오류가 발생하기 쉬운 미적분학(Analytic 그래디언트)이 필요한 
방법입니다.

#### Computing the gradient numerically with finite differences
위에 주어진 공식을 사용하면 그래디언트를 수치적으로 계산할 수 있습니다. 다음은 기울기를 평가하기 위해 함수 
f, 벡터 x를 사용하고 x에서 f의 기울기를 반환하는 일반 함수입니다. gradient는 함수가 가장 가파른 
증가율을 보이는 방향을 알려주지만 이 방향을 따라 얼마나 큰 보폭으로 움직여야하는지는 알려 주지 않습니다. 
이러한 보폭의 크기(학습 속도)를 선택하는 것은 신경망 훈련에서 가장 중요한 하이퍼 파라미터 설정 중 
하나입니다.

**효율성의 문제**
예제에서 총 10 X 3073개의 매개변수를 가지고 있었기 때문에 그래디언트를 평가하고 단일 parameter 
업데이트 수행을 위해 손실 함수의 30731개 평가를 수행해야 했습니다. 하지만 변수가 수천만개에 달한다면 
계산하는 데에 정말 많은 비용이 들것입니다. 개선이 필요하겠죠.

#### Computing the gradient analytically with Calculus

Numerical 방식은 finite difference approximation을 이용하여 계산하기 때문에 매우 
심플합니다. 다만 정확한 값은 아니며 approimate 값이라는 점, 컴퓨팅에 매우 많은 비용이 든다는 아쉬운 
점이 있습니다. 그래서 두 번째 방식으로는 Calculus를 이용하는 분석적 방법으로 gradient를 구하게 
됩니다. 이 경우 빠르게 계산을 할 수 있게 됩니다. 빠른만큼 더 많은 오답이 발생할 수 있습니다. 그래서 
정확성을 평가하기 위해 numerical gradient와 결과를 비교하게 되며, 이를 Gradient check라고 
합니다.

<p>
그래디언트 계산에 하나의 에시를 들어 보겠습니다. 하나의 데이터 포인트에 대한 SVM 손실 함수를 사용하겠습니다.
\begin{equation}
L_i = \sum{j \ne y_i}[max(0, {w_j}^T{x_i} - {w_{y_i}}^T{x_i} + \Delta)]
\end{equation}

가중치인 w에 대하여 함수 간에 차를 구할 수 있습니다. 아래 예시는 $$ w{y_y_i} $$에 대한 
gradient를 구해 봤습니다.

\begin{equation}
\nabla{w{y_i}}L_i = -(\sum{j \ne y_i}\mathbb{1}({w_j}^T{x_i} - 
{w{y_i}}^T{x_i} + \Delta > 0)x_i
\end{equation}

$$ \mathbbb{1} $$은 내부 조건이 참이면 1이고 그렇지 않으면 0인 indicator function입니다.
주어진 문장은 정확한 클래스에 해당하는 W의 행에 대한 기울기만을 설명하며, 코드로 구현할 때는 원하는 
마진을 충족하지 못한 클래스의 개수를 간단히 계산하는 것으로 이해하시면 됩니다. 원하는 마진을 충족하지 
못한 클래스의 개수에 따라 데이터 벡터 xi를 해당 개수로 곱하여 스케일링한 값이 기울기가 됩니다. 주어진 
내용은 올바른 클래스에 해당하는 W의 행에 대한 기울기만을 설명합니다. 다른 행에 대한 기울기는 제시되지 
않습니다.

다른 행(j≠yi)에 대한 기울기는 다음과 같습니다:

\begin{equation}
\nabla{w{y_i}}L_i = \mathbb{1}({w_j}^T{x_i} - 
{w{y_i}}^T{x_i} + \Delta > 0)x_i
\end{equation}

</p>



## 참고
http://cs231n.stanford.edu/schedule.html
