import numpy as np
import tensorflow as tf
from tensorflow import keras

#input male datasets
X_test=np.load('gaussian_filter_3.npy')
#input labels111
#y_test=np.load('gene_y_UKBB_male.npy')

#Reshape dataset
X_test= np.expand_dims(X_test, axis=4)

#male model
# def get_model(width=128, height=128, depth=64): #66
#     """Build a 3D convolutional neural network model."""
#     inputs = tf.keras.Input((width, height, depth, 1))
#     x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
#     x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     # x = tf.keras.layers.Dropout(0.3)(x)
#     x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
#     x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Dropout(0.3)(x)
#     x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
#     x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     #x = tf.keras.layers.Dropout(0.3)(x)
#     # x = tf.keras.layers.Conv3D(filters=16, kernel_size=3, activation="relu")(x)
#     # x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
#     # x = tf.keras.layers.BatchNormalization()(x)
#     # x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.GlobalAveragePooling3D()(x)
#     x = tf.keras.layers.Dense(units=16, activation="relu")(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     #x = tf.keras.layers.Dropout(0.3)(x)
#     # x = tf.keras.layers.Dense(units=128, activation="relu")(x)
#     # x = tf.keras.layers.Dropout(0.3)(x)
#     outputs = tf.keras.layers.Dense(units=6, activation="softmax")(x)
#     # Define the model.
#     model = keras.Model(inputs, outputs, name="3dcnn")
#     return model


#female model

def get_model(width=128, height=128, depth=64): #66
    """Build a 3D convolutional neural network model."""
 
    inputs = tf.keras.Input((width, height, depth, 1))
 
    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
 
    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
 
    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Dropout(0.3)(x)
 
    # x = tf.keras.layers.Conv3D(filters=16, kernel_size=3, activation="relu")(x)
    # x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
 
    # x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(units=64, activation="relu")(x)
    x = tf.keras.layers.Dense(units=32, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Dropout(0.3)(x)
 
    # x = tf.keras.layers.Dense(units=128, activation="relu")(x)
    # x = tf.keras.layers.Dropout(0.3)(x)
 
    outputs = tf.keras.layers.Dense(units=6, activation="softmax")(x)
 
    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


model = get_model(width=80, height=82, depth=96) # female 80x82x96; male 82x86x100
model.summary()
model.load_weights("..\\brain\\saliency map\\saved-model-female.h5") #male


def NormalizeData(data): # saliency map code
    return (data - np.min(data)) / (np.max(data) - np.min(data))
    

saliency_map_list=[]
for index in range(len(X_test)):
  images=X_test[index].reshape((1, *X_test[2].shape)) 
  images = tf.Variable(images, dtype=float)

  with tf.GradientTape() as tape:
      pred = model(images, training=False)
      class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
      loss = pred[0][class_idxs_sorted[0]]
      #loss =tf.Variable(1.122, dtype=float)
  grads = tape.gradient(loss, images)
  dgrad_abs = tf.math.abs(grads)
  dgrad_max_ = np.max(dgrad_abs, axis=4)[0]

  # cut regions
  buf=dgrad_max_
  buf_test=X_test[index].reshape((80,82,96)) # female 80x82x96; male 82x86x100
  for i in range(96):
    for m in range(80):
      for n in range(82):
        if buf[m][n][i]>0 and buf_test[m][n][i]<=0:
          buf[m][n][i]=0
  buf=NormalizeData(buf)
  saliency_map_list.append(buf)
  if index % 100 == 0:
    print(index)
  if index % 500 == 499:
    np.save("saliency_map2_gaussian_3_"+str(index+1)+".npy", saliency_map_list)
    saliency_map_list = []

if saliency_map_list:
  np.save("saliency_map2_gaussian_3_"+str(index+1)+".npy", saliency_map_list)