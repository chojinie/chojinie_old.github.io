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


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic46.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic47.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>    
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic48.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div> 
</div>


하지만, 만약 고양이 사진을 다른 각도에서 찍거나 조명이 달라지거나 한다면 밝기 성분은 당연히 달라질 수 밖에 없겠죠. 사람은 같은 고양이로 인식할 것입니다. 누워있던 숨어있던 뒤돌아 있던, 하지만 컴퓨터는 어렵죠.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic49.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic50.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>    
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic51.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic52.png" class="img-fluid rounded z-depth-1" zoomable=true %}
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

KNN(K-Nearest Neighbor)분류기입니다. task 수행을 위해 모델이 필요하다고 했었죠. 하지만 KNN의 경우 특별한 모델이 필요한 것은 아니고 training data(학습 데이터)의 정답을 모두 저장해두었다가 나중에 test data(테스트 데이터)를 모델에 넣었을 때 정답과 가장 유사한 결과를 예측해내어 분류 task를 수행하는 것이 전부입니다.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic55.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

고양이라는 (Label)을 달고 있는 training data를 저장해두었다가, 나중에 query로 들어오는 Test data와의 거리 비교(Distance Metric)를 하여 그 차이가 가장 작은 데이터를 고양이라고 분류하는 거죠.

Distance Metric 즉, 거리 차이를 비교하는 방법은 대표적으로 두가지가 있습니다. L1 distance와 L2 distance 입니다.

L1 distance: $$ d_1(I_1, I_2) = \sum\limits_{p}|I_1^{p} - I_2^{p}| $$
