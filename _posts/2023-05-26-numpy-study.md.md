---
layout: post
title: Numpy study
date: 2023-05-26
description: recording_a Summary of Numpy
tags: numpy math
categories: study math
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

### array 인덱싱 이해하기

#### 기본 인덱싱

Numpy array의 꽃은 인덱싱입니다. arr 라는 array가 있을 때, arr[5]와 같이 특정한 인덱스를 명시할 수도 있고, arr[5:8]과 같이 범위 형태의 인덱스를 명시할 수도 있습니다.<br> 
arr[:]의 경우 해당 array의 전체 성분을 모두 가져옵니다.<br>

이러한 방식은 2차원 array에 대해서도 유사한 방식으로 적용됩니다. arr2라는 2차원 array를 정의한 뒤,
arr2[2, :]를 실행하면, arr2에서 인덱스가 '2'에 해당하는 행(3행)의 모든 성분이 1차원 array의 형태로 얻어집니다.<br>
arr2[:, 3]을 실행하면 arr2에서 인덱스가 '3'에 해당하는 열(4열)의 모든 성분이 1차원 array의 형태로 얻어집니다.<br>

2차원 array는 이렇게 두개의 인덱스를 받을 수 있는데, ","를 기준으로 앞부분에는 행의 인덱스가 뒷부분에는 열의 인덱스가 입력됩니다.
arr2[1:3, :] 혹은 arr2[:, :2]와 같이, 행 또는 열에 범위 인덱스를 적용하여 여러 개의 행 혹은 열을 얻을 수도 있습니다.<br>

한편 2차원 array에서 4행 3열에 위치한 하나의 성분을 얻고자 할 때는 arr2[3,2]를 실행하면 됩니다.
인덱싱을 통해 선택한 성분에 새로운 값을 대입하는 경우에도, arr2[:2, 1:3] = 0 과 같이 입력값을 넣으면 됩니다.<br>

### NumPy 어레이 정렬 (np.argsort)
#### 기본 사용(오름차순 정렬)

<!--코드블럭-->
- Input
```
import numpy as np

a = np.array([1.5, 0.2, 4.2, 2.5])
s = a.argsort()

print(s)
print(a[s])
```
- Output
```
[1 0 3 2]
[0.2 1.5 2.5 4.2]
```

a는 정렬되지 않은 숫자들의 어레이입니다.
a.argsort()는 어레이 a를 정렬하는 인덱스의 어레이 [1 0 3 2]를 반환합니다.
a[s]와 같이 인덱스의 어레이 s를 사용해서 어레이 a를 다시 정렬하면,
오름차순으로 정렬된 어레이 [0.2 1.5 2.5 4.2]가 됩니다.

#### 내림차순 정렬

- Input
```
import numpy as np

a = np.array([1.5, 0.2, 4.2, 2.5])
s = a.argsort()

print(s)
print(a[s[::-1]])
print(a[s][::-1])
```
- Output
```
[1 0 3 2]
[4.2 2.5 1.5 0.2]
[4.2 2.5 1.5 0.2]
```
<p>
내림차순으로 정렬된 어레이를 얻기 위해서는
a[s[::-1]]와 같이 인덱스 어레이를 뒤집어서 정렬에 사용하거나,
a[s][::-1]과 같이 오름차순으로 정렬된 어레이를 뒤집어주면 됩니다.
</p>


### Reference

https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
<br>
