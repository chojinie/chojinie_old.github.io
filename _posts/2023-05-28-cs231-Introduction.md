---
layout: post
title: cs231n Summary - Introduction
date: 2023-05-28
description: recording_a Summary of lecture
tags: cs231 study AI
categories: study cs231 AI
related_posts: True
giscus_comments: true
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

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic8.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic7.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


#### cs231n Overview

##### Deep Learning Basics - Image Classification
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic35.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
이미지 상의 대상을 무엇으로 분류할지를 classification task라고 합니다. 다양한 방법으로 이를 구현할 수 있습니다.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic36.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic37.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic38.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

선분의 좌측이면 고양이 우측이면 강아지와 같은 방식으로 Classifier을 구현할 수 있습니다. 모델의 정확성을 높이기 위하여 Regularization과 Optimization을 수행하기도 합니다. Neural Network방식의 Classifier도 존재합니다.

##### Perceiving and Understanding the Visual World

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic39.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
컴퓨터가 사물을 분류하는 등 역할을 수행하기 위해서는 결국 사물을 "인지"하고 "이해"하는 과정을 거쳐야 합니다.
사람에게는 정말 쉬운 작업이지만 컴퓨터에게는 이를 위해 다양한 task를 수행해야하며 여기에 맞는 모델이 필요합니다.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic40.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic41.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic42.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/cs231n/assignment1/pic43.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
