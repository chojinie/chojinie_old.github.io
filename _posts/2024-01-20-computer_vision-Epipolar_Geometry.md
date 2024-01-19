---
layout: post
title: computer_vision - Epipolar_Geometry
date: 2024-01-20
description: recording_a Summary of lecture
tags: cv study
categories: study cv
related_posts: True
giscus_comments: true
toc:
  sidebar: left
---

## Epipolar Geometry
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="/assets/img/epipolar.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

CV를 복습하기 위해 작성하는 글.
Structure from Motion(SfM)을 다시 정리하고자 한다. 해당 알고리즘은 여러 단계의 기법들을 이해해야 하는데,
정의나 이런건 다른 포스트에서 기법들을 한대 모아서 설명하도록하고, 여기서는 두 이미지 상의 포인트들의 상관 관계를 정의하는 *Fundamental Matrix* 를 이해하기 '위한' Epipolar Geometry를 정리한다.


### 설명 
<p>
(1)상기 이미지에서 3D-space 상의 한 점 \mathbf{X} 을 양 쪽 카메라의 중심 \mathbf{C} , \mathbf{C\prime} 에서 봤을 때, 이미지 평면에 보이는 점은 각각 \mathbf{x}와 \mathbf{x\prime} 라고 하자. 여기서 \mathbf{x}와 \mathbf{x\prime} 간의 관계를 수식적으로 나타내는 것을 epipolar geometry라고 한다. 즉, 둘 중 한 점만 알더라도 어떠한 기하학적 관계에 따라 나머지 점의 위치를 정확히 예측할 수 있는 것이다. </p>

(2) 그림에서 볼 수 있는 모든 점 x, x', X, C, C'은 모두 같은 평면에 있는 관계이다 (coplanar에 점들이 존재한다고 표현할 수 있음) 그리고 아래와 같은 관계식으로 표현할 수 있다. 해당 평면은 \pi라고 정의할 수 있다. 
'''
\begin{equation}
\vec{Cx} \cdot (\vec{CC'} \times \vec{C\prime x\prime}) = 0
\end{equation}
'''

모든 점들은 coplanar에 있기 때문에, 각 x, x'에서 ray back-projection을 할 경우 X에서 교차한다는 것을 알 수 있고, 이 특성이 점들 간의 상관 관계를 찾게 해주는 강력한 constraint로 작용한다.




## 참고
https://cmsc426.github.io/sfm/<br>
