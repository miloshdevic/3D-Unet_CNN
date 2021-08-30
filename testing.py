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


file = '/Volumes/LaCie_SSD/downsampled_patients_data_training/images/127.npy'
mask = '/Volumes/LaCie_SSD/downsampled_patients_data_training/masks/127.npy'

model = tf.keras.models.load_model('/Users/miloshdevic/Documents/Internship Summer '
                                   '2021/2D_U-Net_CNN/model_for_breast_cancer/',
                                   custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
new_array = np.load(file)
mask_np = np.load(mask).astype('int64')
new_array = interpolation.zoom(new_array, [0.5, 0.5, 0.5])
mask_np = interpolation.zoom(mask_np, [0.5, 0.5, 0.5])

exp_new_array = tf.expand_dims(new_array, axis=-1)
exp_new_array = tf.expand_dims(exp_new_array, axis=0)
prediction = model.predict(exp_new_array)
print(prediction.shape)
print(np.max(np.argmax(prediction, axis=1)))
print(np.max(prediction))
print(type(prediction))
prediction = np.reshape(prediction, (64, 64, 64), order='C')
print(type(prediction))

print("result:", (2 * (mask_np * prediction).sum() + 0.0001) / (mask_np.sum() + prediction.sum() + 0.0001))

fig, ax = plt.subplots(1, 3)
ax[0].imshow(new_array[22])
ax[1].imshow(prediction[22])
ax[2].imshow(mask_np[22])
# plt.imshow(prediction[31])
plt.show()
