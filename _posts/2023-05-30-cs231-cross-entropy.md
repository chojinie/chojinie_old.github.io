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

컴퓨터가 고양이 사진을 본다면 밝기 성분을 [0, 255] 범위의 정수의 집합으로 표현할 것입니다. 예시의 고양이 사진의 전체사이즈는 대략 가로 세로 1:2 비율이라고 본다면, 200px X 400px크기의 이미지로 둘 수 있습니다. 또한 color이미지이므로 RGB 3channels로 구성되어 있습니다. 네모난 창문(window)을 보면 왼쪽 상단에서 우측 상단으로 탐색하면서 해당 위치는 RGB가 각각 얼마만큼 (0~255 사이) 성분이 있는지를 나타내는 값으로 저장됩니다.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/rgb.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

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