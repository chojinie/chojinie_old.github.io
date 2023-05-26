---
layout: post
title: Numpy study
date: 2023-05-23
description: recording_a Summary of Numpy
tags: numpy
categories: study
related_posts: false
toc:
  sidebar: left
---

## numpy.random.choice(a, size=None, replace=True, p=None)

a = 1차원 배열 또는 정수 (정수인 경우, np.arrange(a)와 같은 배열 생성)
size = 정수 또는 튜플(튜플인 경우, 행렬로 리턴됨. (x, y, z) >> x*y*z)
replace = 중복 허용 여부, boolean
p = 1차원 배열, 각 데이터가 선택될 확률 

e.g. np.random.choice(5, 3)
array([0, 3, 4])

0 이상 5미만인 정수 중 3개를 출력(replace=True가 default로 숨어져 있으므로 중복 허용)

np.random.choice(5, 3, replace=False)
array([3, 1, 0])
0 이상 5미만인 정수 중 3개를 출력(replace=False이므로 중복 불가)

## flatnonzero

import numpy as np
a = np.array([1.2, -1.3, 2.1, 5.0, 4.7])
print(np.flatnonzero(a>2)) *# [2 3 4]*

2보다 큰 원소의 index를 array로 리턴.


## np.reshape(X_train, (X_train.shape[0], -1))

np.reshape: It is a function from the NumPy library that allows you to reshape an array.

X_train: It is the array that you want to reshape.

(X_train.shape[0], -1): X_train이 목표로 하는 모양을 의미합니다.
X_train.shape[0]: X_train array의 row의 수를 의미합니다. X_train.shape[0]을 이용하여 reshape된 array에서 row의 숫자가 바뀌지 않고 유지되게 됩니다.
-1: 이것은 NumPy가 제공된 행 수와 원래 배열의 전체 크기를 기반으로 재구성된 배열의 열 수를 자동으로 결정하도록 지시하는 자리 표시자 값입니다.
 -1은 차원이 자동으로 추론되고 조정되어야 함을 나타냅니다. 요약하면, 이 코드 라인은 X_train 배열의 형태를 재구성하며, 행 수는 유지되고 원본 배열의 크기에 따라 열 수가 자동으로 결정됩니다.


## Reference

https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
<br>