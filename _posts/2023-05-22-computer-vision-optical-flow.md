---
layout: post
title: Computer Vision_Optical Flow
date: 2023-05-22
description: recording_a Summary of a optical flow
tags: optical_flow computer_vision Lucas_Kanade Horn_schunck
categories: study
related_posts: false
toc:
  sidebar: left
---

석사 과정 재학 중, 공부한 내용을 기록하기 위해 작성하는 글입니다.

## Optical Flow의 정의

옵티컬 플로우는 관찰자와 장면 간의 상대적인 움직임으로 인해 발생하는 물체, 표면 및 가장자리의 두드러지는 움직임 패턴을 의미합니다.[1] 제가 이해한대로 표현하자면 영상(Image)이나 동영상에서 사물이나 배경의 움직임을 표현하는 기법인데, 사물이나 광원이 움직일 경우 인간의 눈에는 어떠한 흐름의 크기와 방향성이 느껴집니다. 이를 수식적으로 표현하는 것을 뜻합니다.

## Motion Field와 Optical Flow

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/of1.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/of2.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Ref : First Principle of Computer Vision [2]
</div>

$$ \mathrm{v}_i $$(Image velocity) 와 $$ \mathrm{v}_o $$(Seen velocity) 간의 관계를 나타내는 것이 초기 목적입니다. 
$$ \Rightarrow $$ Perspective projection을 이용하게 됩니다.

$$ \mathrm{v}_i $$ 나 Motion Field를 측정할 수 없으므로 Brightness Pattern을 측정할 수 밖에 없습니다.
연속되는 두 이미지 간에 포인트의 모션은 두 포인트의 Depth와 관련이 있습니다.
오른쪽 그림에서 Pinhole은 Z 평면을 가지고 있습니다. 삼각형 모양 간의 비례식으로 표현하여 식을 구할 수 있게 됩니다.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/of3.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

위에서 Motion Field는 측정이 어려우며, Brightness Pattern을 측정할 수 밖에 없다고 했습니다.
오른쪽 그림의 벡터의 길이(크기)는 시간 내에 얼마나 빨리 움직였는지를 나타내며, 화살표의 방향은 어느 방향으로 이동하는지를 나타내줍니다. 
이상적으로는 optical flow는 motion field와 같게 표시 될 수 있습니다. 하지만 현실 세계에서 많은 경우 그 둘은 같지 않게 됩니다.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/of4.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

위 두 그림의 공통점은 광원(Source)과 구체(Sphere)가 있다는 것입니다. 구체는 Lambertian BRDF라고 생각하면 좋을 것 같습니다. 이에 대해서는 추후 보완 설명하겠습니다.
왼쪽의 그림은 광원의 위치는 그대로 두고, 구체가 회전하는 상태입니다. 즉, Motion Field는 존재하지만 Optical Flow는 발생하지 않게 됩니다.
오른쪽 그림은 이와 반대입니다. 구체는 가만히 있지만 광원이 움직여서 밝아 보이는 위치에 변화가 발생합니다. 

##

## Reference

[1] Optical flow. (2023, April 29). In Wikipedia. https://en.wikipedia.org/wiki/Optical_flow

[2] https://youtube.com/@firstprinciplesofcomputerv3258