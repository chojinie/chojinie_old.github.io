---
layout: post
title: cs231n Summary - Introduction
date: 2023-05-28
description: recording_a Summary of lecture
tags: cs231 study AI
categories: study cs231 AI
related_posts: True
toc:
  sidebar: left
---

### 강의 목표

아래 이미지에서 교집합 부분에 해당하는 부분을 학습합니다.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    
</div>


### Agenda
#### 컴퓨터 비전과 딥러닝의 기원의 요약

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic3.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

카메라와 컴퓨터가 개발되면서, 어떻게 하면 컴퓨터가 사람과 같이 vision을 가질 수 있을지 연구하게 되었습니다.
이를 Computer Vision 분야라고 합니다.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic4.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic5.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
Hubel and Wiesel, 1959의 연구를 보면 생명체가 사물을 인식할 때 패턴에 따라 특정한 신호가 나오는 것을
알게 되었고 이를 컴퓨터에 적용하면 어떻게 될까에서 시작한 것 같습니다. 고전적인 Computer Vision
분야에서는 특징점을 찾아내서 사물을 구분하기도 했습니다.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic8.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic7.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
사람과 물체를 인식할 때 부분부분별로 따로 떼어서 인식하기도 했습니다. 또한, 인간이 사물을 파악할 때 edge(가장자리)로 구분 짓는다는 특징을 이용하여 edge detection 분야가 연구되기도 했습니다.
