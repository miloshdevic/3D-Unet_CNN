import tensorflow as tf
import numpy as np
from scipy.ndimage import interpolation
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K


def dice_coef(y_true, y_pred, smooth=1e-8):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    # intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2 * K.sum(y_true * y_pred) + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

# def weighted_bce_seb(y_true, y_pred):
#   weights = y_true * class_weight[1] + (1-y_true)*class_weight[0]
#   bce = K.binary_crossentropy(y_true, y_pred)
#   weighted_bce = K.mean(bce * weights)
#   return weighted_bce
#
# weight_mangrove = np.sum(Y_train==0)/np.sum(Y_train==1)
#
# class_weight = {0: 1.0,
#                 1: weight_mangrove}

def weighted_bce(y_true, y_pred):
  weights = (y_true * 59.) + 1.
  bce = K.binary_crossentropy(y_true, y_pred)
  weighted_bce = K.mean(bce * weights)
  return weighted_bce

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


file = '/Volumes/LaCie_SSD/downsampled_patients_data_training/images/83.npy'
mask = '/Volumes/LaCie_SSD/downsampled_patients_data_training/masks/83.npy'

# model = tf.keras.models.load_model('/Users/miloshdevic/Documents/Internship Summer '
#                                    '2021/2D_U-Net_CNN/model_for_breast_cancer/')#,
                                   #custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})



# img_test = np.load(file)
# img_test = (img_test - np.min(img_test)) / (np.max(img_test) - np.min(img_test))
# mask_test = np.load(mask)
# print("max", np.max(img_test))
# print("min", np.min(img_test))
# fig, ax = plt.subplots(1)
# ax.imshow(mask_test[40])
# # ax.imshow(img_test[22])
# # plt.imshow(prediction[31])
# plt.show()
# input()

model = tf.keras.models.load_model('generator_test_1_no_augmentation.h5',
                                   custom_objects={'weighted_bce': weighted_bce})
# model = tf.keras.models.load_model("/Users/miloshdevic/Documents/Internship Summer 2021/2D_U-Net_CNN/cc_results/",
#                                    custom_objects={'f1': f1_m})
new_array = np.load(file)
new_array = (new_array - np.min(new_array)) / (np.max(new_array) - np.min(new_array))
print(new_array.shape)
mask_np = np.load(mask).astype('int64')
print(mask_np.shape)
new_array = interpolation.zoom(new_array, [1, 1, 1])
print(new_array.shape)
mask_np = interpolation.zoom(mask_np, [1, 1, 1])
print(mask_np.shape)


exp_new_array = np.expand_dims(new_array, axis=-1)
print(exp_new_array.shape)
exp_new_array = np.expand_dims(exp_new_array, axis=0)
print(exp_new_array.shape)

prediction = model.predict(exp_new_array, batch_size=1)
print(prediction.shape)
print(np.max(np.argmax(prediction, axis=1)))
print(np.max(prediction))
print(type(prediction))
prediction = np.reshape(prediction, (128, 128, 128), order='C')
print(type(prediction))

print("result:", (2 * (mask_np * prediction).sum() + 0.0001) / (mask_np.sum() + prediction.sum() + 0.0001))

fig, ax = plt.subplots(1, 3)
ax[0].imshow(new_array[50])
ax[2].imshow(prediction[50])
ax[1].imshow(mask_np[50])
# plt.imshow(prediction[31])
plt.show()
