import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import sys
import random
import tqdm
import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,
                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D, Concatenate)
# from tensorflow.keras.applications import EfficientNetB0
from keras import metrics
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

"""
이미지 데이터셋 Train,Valid 분리
데이터셋은 한 경로 내에 카테고리별 폴더로 나뉘어야 한다.

as_is_path = 이미지 원본 경로
train_path = 트레이닝 세트 생성 경로
valid_path = 테스트 세트 생성 경로
split_size = 분할 비율
"""
def train_val_split(as_is_path, train_path, valid_path, split_size=0.7):

    # 기존 데이터 카테고리 폴더 확인
    # 카테고리 저장
    fold_ = os.listdir(as_is_path)
    fold_.remove('.DS_Store')

    # train set 폴더 생성
    if os.path.isdir(train_path) == False:
        os.mkdir(train_path)
    for i in fold_:
        if os.path.isdir(train_path + '/' + i) == False:
            os.mkdir(train_path + '/' + i)

    # validation set 폴더 생성
    if os.path.isdir(valid_path) == False:
        os.mkdir(valid_path)
    for i in fold_:
        if os.path.isdir(valid_path + '/' + i) == False:
            os.mkdir(valid_path + '/' + i)

    # split 실행 및 copy하여 적재
    for idx in fold_:
        listfile = os.listdir(as_is_path + '/' + idx)
        train_size = int(len(listfile)*split_size)
        print("{} | train {}, valid {} split".format(idx, train_size, len(listfile)-train_size))
        for fname in listfile[0:train_size]:
            shutil.copyfile(os.path.join(as_is_path + '/' + idx, fname),os.path.join(train_path + '/' + idx, fname))
        for fname in listfile[train_size:]:
            shutil.copyfile(os.path.join(as_is_path + '/' + idx, fname),os.path.join(valid_path + '/' + idx, fname))

# 데이터셋 경로
base_path = '/Users/dkcns/PycharmProjects/Image_Multilabel/dataset/'
as_is_path = base_path + 'balance'
train_path = as_is_path + '_trd'
valid_path = as_is_path + '_valid'
test_path = base_path + 'balance_test'

# train_val_split 함수 실행
train_val_split(as_is_path, train_path, valid_path, split_size=0.8)

# 전처리(train)
# 이미지 스케일링
trainDataGen = ImageDataGenerator(rescale=1./255, # 이미지 스케일링
                                 rotation_range = 30, # 임의로 이미지를 회전
                                 width_shift_range=0.1, # 임의로 이미지를 수평 이동
                                 height_shift_range=0.1, # 임의로 이미지를 수직 이동
                                 shear_range=0.2, # 임의로 이미지를 변형
                                 zoom_range=0.2, # 임의로 이미지를 확대/축소
                                 horizontal_flip=False, # 수평방향으로 뒤집기
                                 fill_mode='nearest' # 이미지 경계의 바깥 공간을 어떻게 채울지에 대한 파라미터
                                 )

testDataGen = ImageDataGenerator(rescale=1./255)

from glob import glob

"""
MultiLable을 만들기 위한 함수
여기서는 앞에가 color, 뒤에가 item인 black_shirt.png 로 구성되어있기에 목표에 맞게 작업
향후 변경 필요시 columns= / get_dummies 등을 변경하여 사용
"""


def make_multi_lable(data_path):
    data_set = glob(data_path + '/**/*.*', recursive=True)
    dt_f = pd.DataFrame(list(map(lambda x: x.split('/')[-2].split('_'), data_set)), columns=['color', 'item'])
    dt_f = pd.concat([pd.get_dummies(dt_f['color']), pd.get_dummies(dt_f['item'])], axis=1)
    dt_f_col = list(dt_f.columns)
    dt_f['path'] = data_set
    dt_f

    return dt_f, dt_f_col

# 이미지 스케일링
dt_train, dt_train_col = make_multi_lable(train_path)
dt_valid, dt_valid_col = make_multi_lable(valid_path)

# Train Set
trainGenSet = trainDataGen.flow_from_dataframe(
      dt_train,
      x_col='path',
      y_col=dt_train_col,
      batch_size=32,
      seed=42,
      shuffle=True,
      target_size=(100,100),
      class_mode='raw')

# Valid Set
testGenSet = testDataGen.flow_from_dataframe(
      dt_valid,
      x_col='path',
      y_col=dt_valid_col,
      batch_size=32,
      seed=42,
      shuffle=True,
      target_size=(100,100),
      class_mode='raw')

# Test Set
# 귀찮아서 Valid Set 그대로 셔플 제외하여 사용
finalGenSet = testDataGen.flow_from_dataframe(
      dt_valid,
      x_col='path',
      y_col=dt_valid_col,
      batch_size=32,
      seed=42,
      shuffle=False,
      target_size=(100,100),
      class_mode='raw')

# White Pants
whiteGenSet = testDataGen.flow_from_directory(
    test_path,
    target_size=(100,100),
    batch_size=32,
    class_mode=None
)

# 모델
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), padding='same', input_shape=(100,100,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(7, activation='softmax'))

# 학습
model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

# fig_generator
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2000)
checkpoint = ModelCheckpoint('/Users/dkcns/PycharmProjects/Image_Multilabel/model/multilabel_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model.fit_generator(
    trainGenSet,
    steps_per_epoch=43,
    epochs=50,
    validation_data=testGenSet,
    validation_steps=10,
    callbacks=[early_stopping, checkpoint]
)

scores = model.evaluate_generator(testGenSet)
print(scores)

model_ = load_model('/Users/dkcns/PycharmProjects/Image_Multilabel/model/multilabel_model.h5')
pred = model_.predict_generator(finalGenSet, steps=1)
pred_df = pd.DataFrame(pred, columns = dt_valid_col)
pred_df['path'] = dt_valid['path'][0:32]

import matplotlib.pyplot as plt
j = 3
a = plt.imread(pred_df['path'][j])
plt.imshow(a)
pred_df.iloc[j:j+1,:] * 100

pred = model_.predict_generator(whiteGenSet, steps=1)
pred_df = pd.DataFrame(pred, columns = dt_valid_col)
print(pred_df*100)
a = plt.imread(glob(test_path + '/test/*')[0])
plt.imshow(a)