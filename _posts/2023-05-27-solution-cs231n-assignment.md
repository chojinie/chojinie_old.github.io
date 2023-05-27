---
layout: post
title: cs231n assignment solution
date: 2023-05-27
description: recording_a Summary of assignment
tags: cs231 study AI
categories: study cs231 AI
related_posts: false
toc:
  sidebar: left
---

## Assignment 1
### Inline Question 1
Notice the structured patterns in the distance matrix, where some rows or columns are visibly brighter. 
(Note that with the default color scheme black indicates low distances while white indicates high distances.)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/assignmentinlinequestion1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

- What in the data is the cause behind the distinctly bright rows?
- What causes the columns?

$\color{blue}{\textit Your Answer:}$ *fill this in.*

<p>
훈련 데이터 세트에 포함되지 않은 클래스로부터의 관측이거나, 적어도 대부분의 훈련 데이터와 매우 다른 관측일 가능성이 높습니다. 아마도 배경 색상과 관련하여 큰 차이가 있을 것입니다.
학습 데이터 포인트가 테스트 데이터 내의 어떠한 포인트와도 비슷하지 않음을 의미합니다.
</p>
