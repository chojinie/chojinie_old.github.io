---
layout: post
title: Numpy study
date: 2023-05-26
description: recording_a Summary of Numpy
tags: numpy
categories: study
related_posts: false
toc:
  sidebar: left
---

### numpy.random.choice(a, size=None, replace=True, p=None)

a = 1차원 배열 또는 정수 (정수인 경우, np.arrange(a)와 같은 배열 생성)<br>
size = 정수 또는 튜플(튜플인 경우, 행렬로 리턴됨. (x, y, z) >> x*y*z)<br>
replace = 중복 허용 여부, boolean<br>
p = 1차원 배열, 각 데이터가 선택될 확률<br>

e.g. np.random.choice(5, 3)<br>
array([0, 3, 4])<br>

0 이상 5미만인 정수 중 3개를 출력(replace=True가 default로 숨어져 있으므로 중복 허용)<br>

np.random.choice(5, 3, replace=False)<br>
array([3, 1, 0])<br>
0 이상 5미만인 정수 중 3개를 출력(replace=False이므로 중복 불가)<br>

### flatnonzero

import numpy as np<br>
a = np.array([1.2, -1.3, 2.1, 5.0, 4.7])<br>
print(np.flatnonzero(a>2)) *# [2 3 4]*<br>

2보다 큰 원소의 index를 array로 리턴.


### np.reshape(X_train, (X_train.shape[0], -1))

np.reshape: array를 원하는 형태로 reshape할 수 있게 해주는 라이브러리.

X_train: reshape하고 싶은 array<br>
(X_train.shape[0], -1): X_train이 목표로 하는 모양을 의미합니다.<br>
X_train.shape[0]: X_train array의 row의 수를 의미합니다. <br>
X_train.shape[0]을 이용하여 reshape된 array에서 row의 숫자가 바뀌지 않고 유지되게 됩니다.<br>
-1: 이것은 NumPy가 제공된 행 수와 원래 배열의 전체 크기를 기반으로 재구성된 배열의 열 수를 자동으로 결정하도록 지시하는 자리 표시자 값입니다.<br>
 -1은 차원이 자동으로 추론되고 조정되어야 함을 나타냅니다. 요약하면, 이 코드 라인은 X_train 배열의 형태를 재구성하며, 행 수는 유지되고 원본 배열의 크기에 따라 열 수가 자동으로 결정됩니다.<br>

### np.zeros

numpy.zeros(shape, dtype=float, order='C', *, like=None)<br>
Return a new array of given shape and type, filled with zeros.<br>

Parameters<br>
shape : int or tuple of ints<br>
np.zeros((2, 1))<br>
array([[ 0.],<br>
       [ 0.]])<br>

dtype : data-type, optional<br>
The desired data-type for the array, e.g., numpy.int8. Default is numpy.float64.<br>

order : {‘C’, ‘F’}, optional, default: ‘C’<br>
Whether to store multi-dimensional data in row-major (C-style) or column-major (Fortran-style) order in memory.<br>

like : array_like, optional<br>
Reference object to allow the creation of arrays which are not NumPy arrays. If an array-like passed in as like supports the __array_function__ protocol, the result will be defined by it. In this case, it ensures the creation of an array object compatible with that passed in via this argument.

### Reference

https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
<br>