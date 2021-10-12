#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=redefined-outer-name
# pylint: disable=g-bad-import-order
"""Build and train neural networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os
from data_load import DataLoader

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from numpy.random import seed


# In[ ]:


logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# In[ ]:


# Function: Convert some hex value into an array for C programming
def hex_to_c_array(hex_data, var_name):

  c_str = ''

  # Create header guard
  c_str += '#ifndef ' + var_name.upper() + '_H\n'
  c_str += '#define ' + var_name.upper() + '_H\n\n'

  # Add array length at top of file
  c_str += '\nconst unsigned int g_magic_wand_model_data_size = ' + str(len(hex_data)) + '; '

  # Declare C variable
  c_str += 'alignas(16) const unsigned char g_magic_wand_model_data[] = {'
  hex_array = []
  for i, val in enumerate(hex_data) :

    # Construct string from hex
    hex_str = format(val, '#04x')

    # Add formatting so each line stays within 80 characters
    if (i + 1) < len(hex_data):
      hex_str += ','
    if (i + 1) % 12 == 0:
      hex_str += '\n '
    hex_array.append(hex_str)

  # Add closing brace
  c_str += '\n ' + format(' '.join(hex_array)) + '\n};\n\n'

  # Close out header guard
  c_str += '#endif //' + var_name.upper() + '_H'

  return c_str


# In[ ]:


def reshape_function(data, label):
  reshaped_data = tf.reshape(data, [-1, 3, 1])
  return reshaped_data, label


# In[ ]:


def calculate_model_size(model):
  print(model.summary())
  var_sizes = [
      np.product(list(map(int, v.shape))) * v.dtype.size
      for v in model.trainable_variables
  ]
  print("Model size:", sum(var_sizes) / 1024, "KB")


# In[ ]:


def build_cnn(seq_length):
  """Builds a convolutional neural network in Keras."""
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(
          8, (4, 3),                                      # 8 Filter a 4x3
          padding="same",
          activation="relu",
          input_shape=(seq_length, 3, 1)),                # output_shape=(batch, 128, 3, 8) // seq_length = 128 im Code def
      tf.keras.layers.MaxPool2D((3, 3)),                  # (batch, 42, 1, 8)  // 3,3 d.h. drittelt Höhe, Breite
      tf.keras.layers.Dropout(0.1),                       # (batch, 42, 1, 8)
      tf.keras.layers.Conv2D(16, (4, 1), padding="same",
                             activation="relu"),          # (batch, 42, 1, 16) // 16 Filter a 4x1
      tf.keras.layers.MaxPool2D((3, 1), padding="same"),  # (batch, 14, 1, 16) // 3,1 d.h. drittelt Höhe, Breite bleibt
      tf.keras.layers.Dropout(0.1),                       # (batch, 14, 1, 16)
      tf.keras.layers.Flatten(),                          # (batch, 224)       // 224 = 14 x 16 x 1
      tf.keras.layers.Dense(16, activation="relu"),       # (batch, 16)
      tf.keras.layers.Dropout(0.1),                       # (batch, 16)
      tf.keras.layers.Dense(4, activation="softmax")      # (batch, 4)
  ])
  model_path = os.path.join("./netmodels", "CNN")
  print("Built CNN.")
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  model.load_weights("./netmodels/CNN/weights.h5")
  return model, model_path


# In[ ]:


"""    def build_rf(seq_length):
          model = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
          model_path = os.path.join("./netmodels", "RF")
          print("Built Random Forest.")
          if not os.path.exists(model_path):
            os.makedirs(model_path)
          return model, model_path     """


# In[ ]:


def build_lstm(seq_length):
  """Builds an LSTM in Keras."""
  model = tf.keras.Sequential([
      tf.keras.layers.Bidirectional(
          tf.keras.layers.LSTM(22),
          input_shape=(seq_length, 3)),  # output_shape=(batch, 44)
      tf.keras.layers.Dense(4, activation="sigmoid")  # (batch, 4)
  ])
  model_path = os.path.join("./netmodels", "LSTM")
  print("Built LSTM.")
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  return model, model_path


# In[ ]:


def load_data(train_data_path, valid_data_path, test_data_path, seq_length):
  data_loader = DataLoader(train_data_path,
                           valid_data_path,
                           test_data_path,
                           seq_length=seq_length)
  data_loader.format()
  return data_loader.train_len, data_loader.train_data, data_loader.valid_len,       data_loader.valid_data, data_loader.test_len, data_loader.test_data


# In[ ]:


def build_net(args, seq_length):
  if args.model == "CNN":
    model, model_path = build_cnn(seq_length)
  elif args.model == "LSTM":
    model, model_path = build_lstm(seq_length)
  elif args.model == "RF":
    model, model_path = build_rf(seq_length)
  else:
    print("Please input correct model name.(CNN  LSTM  RF)")
  return model, model_path


# In[ ]:


def train_net(
    model,
    model_path,  # pylint: disable=unused-argument
    train_len,  # pylint: disable=unused-argument
    train_data,
    valid_len,
    valid_data,  # pylint: disable=unused-argument
    test_len,
    test_data,
    kind):
  
  # This callback will stop the training when there is no improvement in
  # the loss for [patience] consecutive epochs.
  callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50)

  """Trains the model."""
  calculate_model_size(model)
  epochs = 50
  batch_size = 64
  model.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
  if kind == "CNN":
    train_data = train_data.map(reshape_function)
    test_data = test_data.map(reshape_function)
    valid_data = valid_data.map(reshape_function)
  test_labels = np.zeros(test_len)
  idx = 0
  for data, label in test_data:  # pylint: disable=unused-variable
    test_labels[idx] = label.numpy()
    idx += 1
  train_data = train_data.batch(batch_size).repeat()
  valid_data = valid_data.batch(batch_size)
  test_data = test_data.batch(batch_size)
  history = model.fit(train_data,
            epochs=epochs,
            validation_data=valid_data,
            steps_per_epoch=100,
            validation_steps=int((valid_len - 1) / batch_size + 1),
            callbacks=[callback]
           )
  loss, acc = model.evaluate(test_data)
  pred = np.argmax(model.predict(test_data), axis=1)
  confusion = tf.math.confusion_matrix(labels=tf.constant(test_labels),
                                       predictions=tf.constant(pred),
                                       num_classes=4)
  print(confusion)
    
  plt.plot(history.history['accuracy'], label='accuracy')
  plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.ylim([0, 1])
  plt.legend(loc='lower right')
  plt.show()

  print("Loss {}, Accuracy {}".format(loss, acc))
  # Convert the model to the TensorFlow Lite format without quantization
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()

  # Save the model to disk
  open("model.tflite", "wb").write(tflite_model)

  # Convert the model to the TensorFlow Lite format with quantization
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_model = converter.convert()

  # Save the model to disk
  open("model_quantized.tflite", "wb").write(tflite_model)

  basic_model_size = os.path.getsize("model.tflite")
  print("Basic model is %d bytes" % basic_model_size)
  quantized_model_size = os.path.getsize("model_quantized.tflite")
  print("Quantized model is %d bytes" % quantized_model_size)
  difference = basic_model_size - quantized_model_size
  print("Difference is %d bytes" % difference)
  
  tflite_model_name = 'model_quantized'  # Will be given .tflite suffix
  c_model_name = 'model_quantized'       # Will be given .h suffix

  # Write TFLite model to a C source (or header) file
  with open(c_model_name + '.h', 'w') as file:
      file.write(hex_to_c_array(tflite_model, c_model_name))


# In[ ]:


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", "-m")
  parser.add_argument("--person", "-p")
  args = parser.parse_args()
  
  np.random.seed(24)

  print(seed)

  seq_length = 128

  print("Start to load data...")
  if args.person == "true":
    train_len, train_data, valid_len, valid_data, test_len, test_data =         load_data("./person_split/train", "./person_split/valid",
                  "./person_split/test", seq_length)
  else:
    train_len, train_data, valid_len, valid_data, test_len, test_data =         load_data("./data/train", "./data/valid", "./data/test", seq_length)

  print("Start to build net...")
  model, model_path = build_net(args, seq_length)

  print("Start training...")
  train_net(model, model_path, train_len, train_data, valid_len, valid_data,
            test_len, test_data, args.model)

  print("Training finished!")


# In[ ]:




