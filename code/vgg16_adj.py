from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Conv2D, GlobalAveragePooling2D
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping

def vgg16_adj(model_type='pooling',
              loss='binary_crossentropy',
              final_dense=2,
              final_activation='sigmoid'):

    model = Sequential(name="VGG16_ADJ")
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', input_shape=(224,224,3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    if model_type == 'flatten':
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(final_dense, activation=final_activation))
    else:
        model.add(GlobalAveragePooling2D())
        model.add(Dense(512, activation='relu', name='after_gap'))
        model.add(Dense(final_dense, activation=final_activation))
    model.summary()

    # 컴파일
    model.compile(loss=loss,
                 optimizer='adam',
                 metrics=['accuracy'])

    return model

if __name__ == '__main__':

    ##### USE CASE SAMPLE #####

    # 모델 불러오기
    model = vgg16_adj(model_type = 'pooling',       # pooling 또는 flatten 형식 설정
                      loss='binary_crossentropy',   # keras 기준, 필요한 loss function 설정
                      final_dense=7,                # 최종 결과 Dense 설정
                      final_activation='sigmoid')   # 마지막 Activation 설정

    # 제너레이터 설정
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2000)
    checkpoint = ModelCheckpoint(INPUT_MODEL_SAVE_PATH + 'vgg_adj.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # 데이터 불러오기
    INPUT_YOUR_TRAIN_DATASET    # 트레이닝 배치 세트를 넣어주세요
    INPUT_YOUR_VALID_DATASET    # 검증용 배치 세트를 넣어주세요
    INPUT_YOUR_TEST_DATASET     # 테스트 배치 세트를 넣어주세요
    INPUT_MODEL_SAVE_PATH       # 모델을 임시 저장할 경로를 넣어주세요

    # 학습
    model.fit_generator(
        INPUT_YOUR_TRAIN_DATASET,
        steps_per_epoch=43,
        epochs=50,
        validation_data=INPUT_YOUR_VALID_DATASET,
        validation_steps=10,
        callbacks=[early_stopping, checkpoint]
    )

    # 평가
    scores = model.evaluate_generator(INPUT_YOUR_TEST_DATASET)
    print(scores)