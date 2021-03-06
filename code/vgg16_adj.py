from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Conv2D, GlobalAveragePooling2D
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping

def vgg16_adj(model_type='pooling',
              loss='binary_crossentropy',
              final_dense=2,
              final_activation='sigmoid',
              input_shape=(224,224,3)):

    model = Sequential(name="VGG16_ADJ")
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', input_shape=input_shape, activation='relu'))
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

    # ?????????
    model.compile(loss=loss,
                 optimizer='adam',
                 metrics=['accuracy'])

    return model

if __name__ == '__main__':

    ##### USE CASE SAMPLE #####

    # ?????? ????????????
    model = vgg16_adj(model_type = 'pooling',       # pooling ?????? flatten ?????? ??????
                      loss='binary_crossentropy',   # keras ??????, ????????? loss function ??????
                      final_dense=7,                # ?????? ?????? Dense ??????
                      final_activation='sigmoid',   # ????????? Activation ??????
                      input_shape=(224,224,3))      # ????????? input shape ??????(VGG16 ????????? 224,224,3)

    # ??????????????? ??????
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2000)
    checkpoint = ModelCheckpoint(INPUT_MODEL_SAVE_PATH + 'vgg_adj.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # ????????? ????????????
    INPUT_YOUR_TRAIN_DATASET    # ???????????? ?????? ????????? ???????????????
    INPUT_YOUR_VALID_DATASET    # ????????? ?????? ????????? ???????????????
    INPUT_YOUR_TEST_DATASET     # ????????? ?????? ????????? ???????????????
    INPUT_MODEL_SAVE_PATH       # ????????? ?????? ????????? ????????? ???????????????

    # ??????
    model.fit_generator(
        INPUT_YOUR_TRAIN_DATASET,
        steps_per_epoch=43,
        epochs=50,
        validation_data=INPUT_YOUR_VALID_DATASET,
        validation_steps=10,
        callbacks=[early_stopping, checkpoint]
    )

    # ??????
    scores = model.evaluate_generator(INPUT_YOUR_TEST_DATASET)
    print(scores)