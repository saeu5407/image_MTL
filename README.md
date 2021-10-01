# 멀티 태스크 러닝(Multi-task Learning)

> 케라스를 활용해 멀티태스크 러닝을 구현해보고자 하는 repo입니다.

---

### 구현 순서

구현하는 데 앞서 image classification 모델을 각각 `Multiclass`, `Multilabel`로 개발해보며 각각의 차이를 익혀보고,
최종적으로는 이 분류기들을 활용한 `Multi-task Learning` 모델을 구현해보겠습니다.

`Multiclass`, `Multilabel`을 비교하는 데에는 크롤링한 의류 데이터를 활용할 예정이며,
`Multilabel`을 잘 사용할 만한 예제인 Fashion MNIST를 통해 멀티 레이블만 한 번 더 구현해보겠습니다.<br>
마지막으로 다수의 `Multiclass`, `Multilabel` 분류기들을 Hard Parameter Sharing 방법으로 연결한 `Multi-task Learning` 모델을 구현해보겠습니다.

---

### 코드
코드 | 내용 | 데이터
:---|:---|:---
vgg16_adj.py | `VGG16`을 살짝 수정하여 구현한 모델 | 
multiclass_cnn.ipynb | 간단한 데이터셋으로 `Multiclass`모델 구현 | 의류 데이터
multilabel_cnn.ipynb | 간단한 데이터셋으로 `Multilabel`모델 구현 | 의류 데이터
multilable_cnn_fashion_mnist.ipynb | 인식 분야 데이터를 활용하여 `Multilabel`모델 구현 | Fashion MNIST
multitask_learning.ipynb | 4개의 `Multiclass`, 1개의 `Multilabel`을 공유하는 `Multi-task Learning`구현 | 페이트 게임 일러스트 데이터

---

### 데이터
의류 데이터 : [데이터 링크](https://drive.google.com/file/d/1nYeCJ3Vd89ReJHMEM1rwEtg8tp4P-Dd_/view?usp=sharing)</br>
Fashion MNIST : </br>
페이트 게임 일러스트 데이터 : [원작자의 코드 및 데이터 링크](https://github.com/sugi-chan/fgo-multi-task-keras).
