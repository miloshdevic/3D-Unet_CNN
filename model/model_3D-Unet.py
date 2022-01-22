import os
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from Data_generator import DataGenerator

# Hyperparameters
HEIGHT = 128
WEIGHT = 128
DEPTH = 128
BATCH_SIZE = 8


###################################################################################################################
# Different loss functions/methods that have been used

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


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# def weighted_bce(y_true, y_pred):
#   weights = y_true * class_weight[1] + (1-y_true)*class_weight[0]
#   bce = K.binary_crossentropy(y_true, y_pred)
#   weighted_bce = K.mean(bce * weights)
#   return weighted_bce
# 
# 
# weight_mangrove = np.sum(Y_train==0)/np.sum(Y_train==1)
# 
# class_weight = {0: 1.0,
#                 1: weight_mangrove}

###################################################################################################################

# Design model
model = Sequential()

# Architecture
inputs = tf.keras.layers.Input((HEIGHT, WEIGHT, DEPTH, 1), batch_size=BATCH_SIZE)

# go from 128x128x128 to 64x64x64
# filters go from 1 to 32
conv0 = tf.keras.layers.Conv3D(16, (2, 2, 2), padding='same')(inputs)
norm0 = tf.keras.layers.BatchNormalization()(conv0)
leaky0 = tf.keras.layers.LeakyReLU()(norm0)  # (128, 128, 128, 16)
conv05 = tf.keras.layers.Conv3D(32, (2, 2, 2), padding='same')(leaky0)
norm05 = tf.keras.layers.BatchNormalization()(conv05)
leaky05 = tf.keras.layers.LeakyReLU()(norm05)  # (128, 128, 128, 32) --> stays 128 bc of padding
max0 = tf.keras.layers.Conv3D(32, (2, 2, 2), strides=(2, 2, 2))(leaky05)

# from 64x64x64 to 32x32x32
conv1 = tf.keras.layers.Conv3D(32, (2, 2, 2), padding='same')(max0)
norm1 = tf.keras.layers.BatchNormalization()(conv1)
leaky1 = tf.keras.layers.LeakyReLU()(norm1)  # (64, 64, 64, 32)
conv2 = tf.keras.layers.Conv3D(64, (2, 2, 2), padding='same')(leaky1)
norm2 = tf.keras.layers.BatchNormalization()(conv2)
leaky2 = tf.keras.layers.LeakyReLU()(norm2)  # (64, 64, 64, 64) --> stays 64 bc of padding
max1 = tf.keras.layers.Conv3D(64, (2, 2, 2), strides=(2, 2, 2))(leaky2)  # (32, 32, 32, 64)

# bottom layer of U-net
conv3 = tf.keras.layers.Conv3D(128, (2, 2, 2), padding='same')(max1)
norm3 = tf.keras.layers.BatchNormalization()(conv3)
leaky3 = tf.keras.layers.LeakyReLU()(norm3)  # (32, 32, 32, 128)
conv4 = tf.keras.layers.Conv3D(128, (2, 2, 2), padding='same')(leaky3)
norm4 = tf.keras.layers.BatchNormalization()(conv4)
leaky4 = tf.keras.layers.LeakyReLU()(norm4)  # (32, 32, 32, 128)
convt1 = tf.keras.layers.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2))(leaky4)  # (64, 64, 64, 64)

conc1 = tf.keras.layers.Concatenate(axis=-1)([convt1, leaky2])  # (64, 64, 64, 64) + (64, 64, 64, 64)
conv5 = tf.keras.layers.Conv3D(64, (2, 2, 2), padding='same')(conc1)
norm5 = tf.keras.layers.BatchNormalization()(conv5)
leaky5 = tf.keras.layers.LeakyReLU()(norm5)  # (64, 64, 64, 64)
conv6 = tf.keras.layers.Conv3D(64, (2, 2, 2), padding='same')(leaky5)
norm6 = tf.keras.layers.BatchNormalization()(conv6)
leaky6 = tf.keras.layers.LeakyReLU()(norm6)  # (64, 64, 64, 64)
convt2 = tf.keras.layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2))(leaky6)  # (128, 128, 128, 32)

conc2 = tf.keras.layers.Concatenate(axis=-1)([convt2, leaky05])  # (128, 128, 128, 32) + (128, 128, 128, 32)
conv7 = tf.keras.layers.Conv3D(32, (2, 2, 2), padding='same')(conc2)
norm7 = tf.keras.layers.BatchNormalization()(conv7)
leaky7 = tf.keras.layers.LeakyReLU()(norm7)  # (128, 128, 128, 32)
conv8 = tf.keras.layers.Conv3D(32, (2, 2, 2), padding='same')(leaky7)
norm8 = tf.keras.layers.BatchNormalization()(conv8)
leaky8 = tf.keras.layers.LeakyReLU()(norm8)  # (128, 128, 128, 32)

output = tf.keras.layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(leaky8)
checkpointer = tf.keras.callbacks.ModelCheckpoint('~/scratch/outputs/model_for_breast_cancer', verbose=1,  save_best_only=True,
                                                  monitor='val_loss')
# save_weights_only=False,
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=50),
    # 'log_dir' creates folder by the given name and saves everything into that
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    checkpointer
]

optimizer = tf.keras.optimizers.Adam(0.001, 0.9)  # 0.001 is the learning rate

model = tf.keras.Model(inputs=[inputs], outputs=[output])
model.compile(optimizer='adam', loss='binary_crossentropy')  # , metrics=['accuracy', dice_coef])
model.summary()

list_of_images = os.listdir("/home/mdevic31/scratch/data/images/")
list_of_masks = os.listdir("/home/mdevic31/scratch/data/masks/")

# Parameters
params = {'dim': (HEIGHT, WEIGHT, DEPTH),
          'batch_size': BATCH_SIZE,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True}

# Datasets
partition = {'train': list_of_images[0:80], 'validation': list_of_images[80:112]}
labels = {'train': list_of_masks[0:80], 'validation': list_of_masks[80:112]}

# Generators
training_generator = DataGenerator(partition['train'], labels['train'], **params)
validation_generator = DataGenerator(partition['validation'], labels['validation'], **params)

# Train model on dataset
model.fit(training_generator, epochs=1000, batch_size=BATCH_SIZE,  validation_data=validation_generator,
          callbacks=callbacks, verbose=1)
