import os
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.models import Sequential
from Data_generator import DataGenerator


def dice_coef(y_true, y_pred, smooth=1e-8):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    print("types: ", y_true, y_pred)
    # intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2 * K.sum(y_true * y_pred) + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)  # (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


# Design model
model = Sequential()

# Architecture
inputs = tf.keras.layers.Input((64, 64, 64, 1))
# s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

# Contraction path
c1 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c1)

c2 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c2)

c3 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling3D((2, 2, 2))(c3)

c4 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(c4)

c5 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

# Expansive path
u6 = tf.keras.layers.Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4], axis=-1)
c6 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3], axis=-1)
c7 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2], axis=-1)
c8 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=-1)
c9 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(c9)

checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_breast_cancer', verbose=1,  save_best_only=True,
                                                  monitor='val_loss')
# save_weights_only=False,
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=15),
    # 'log_dir' creates folder by the given name and saves everything into that
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    checkpointer
]

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss=dice_coef_loss, metrics=['accuracy', dice_coef])
model.summary()

list_of_images = os.listdir("train_data/all_imgs/")
list_of_masks = os.listdir("train_data/all_msks/")

# Parameters
params = {'dim': (64, 64, 64),
          'batch_size': 2,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True}

# Datasets
partition = {'train': list_of_images[0:10], 'validation': list_of_images[10:12]}
labels = {'train': list_of_masks[0:10], 'validation': list_of_masks[10:12]}
print(list_of_images[10:12])
print(list_of_images[0:10])
# Generators
training_generator = DataGenerator(partition['train'], labels['train'], **params)
validation_generator = DataGenerator(partition['validation'], labels['validation'], **params)

# Train model on dataset
model.fit(training_generator, epochs=20, batch_size=2,
          validation_data=validation_generator, callbacks=callbacks, verbose=1)
