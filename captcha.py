import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from IPython.display import FileLink

img_folder = "../input/russian-captcha-images-base64/translit/images"

EPOCH = 6
TEST_COUNT = 10000
BATCH_SIZE = 16

# ----------------------
df = pd.read_csv('../input/russian-captcha-images-base64/labels.csv', header=None, encoding='utf-8', delimiter=';', names=['text', 'filename'])

data = {row.text: row.filename for row in df.itertuples()}

input = data
characters = sorted(set(''.join(data.keys())))
char_to_num = {v: i for i, v in enumerate(characters)}

num_to_char = {str(i): v for i, v in enumerate(characters)}
num_to_char['-1'] = 'UKN'

print(num_to_char)
# ----------------------
# {'0': '2', '1': '4', '2': '5', '3': '6', '4': '7', '5': '8', '6': '9', '7': 'б', '8': 'в', '9': 'г', '10': 'д', '11': 'ж', '12': 'к', '13': 'л', '14': 'м', '15': 'н', '16': 'п', '17': 'р', '18': 'с', '19': 'т', '-1': 'UKN'}
# ----------------------
def compute_perf_metric(predictions, groundtruth):
    if predictions.shape == groundtruth.shape:
        return np.sum(predictions == groundtruth)/(predictions.shape[0]*predictions.shape[1])
    else:
        raise Exception('Error : the size of the arrays do not match. Cannot compute the performance metric')
# ----------------------
def encode_single_sample(filename):
    img_path = os.path.join(img_folder, filename)
    # Read image file and returns a tensor with dtype=string
    img = tf.io.read_file(img_path)

    try:
      img = tf.io.decode_png(img, channels=3)
    except Exception as e:
      print(img_path)
      raise e

    # Scales and returns a tensor with dtype=float32
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Transpose the image because we want the time dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])

    return img.numpy()

def create_train_and_validation_datasets():
    # Loop on all the files to create X whose shape is (1040, 50, 200, 1) and y whose shape is (1040, 5)
    X, y = [],[]

    items = list(input.items())
    train_dataset = items[:TEST_COUNT] + items[-TEST_COUNT:]
    test_dataset = items[TEST_COUNT:-TEST_COUNT]

    y, X = zip(*train_dataset)

    X = np.asarray(list(map(encode_single_sample, X)))
    y = np.asarray([list(map(lambda x:char_to_num[x], label)) for label in y])

    y = tf.keras.preprocessing.sequence.pad_sequences(y, 7, padding='post', value=-1)

    print(X.shape)
    print(y.shape)

    # Split X, y to get X_train, y_train, X_val, y_val 
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)
    #X_train, X_val = X_train.reshape(936,200,50,1), X_val.reshape(104,200,50,1)
    return X_train, X_val, y_train, y_val, test_dataset
# ----------------------
X_train, X_val, y_train, y_val, test_dataset = create_train_and_validation_datasets()

print(X_train.shape, X_val.shape)
fig=plt.figure(figsize=(20, 10))
fig.add_subplot(2, 4, 1)
plt.imshow(X_train[0], cmap='gray')
plt.title('Image from X_train with label '+ str(y_train[0]))
plt.axis('off')
fig.add_subplot(2, 4, 2)
plt.imshow(X_train[135], cmap='gray')
plt.title('Image from X_train with label '+ str(y_train[135]))
plt.axis('off')
fig.add_subplot(2, 4, 3)
plt.imshow(X_val[0], cmap='gray')
plt.title('Image from X_val with label '+ str(y_val[0]))
plt.axis('off')
fig.add_subplot(2, 4, 4)
plt.imshow(X_val[23], cmap='gray')
plt.title('Image from X_val with label '+ str(y_val[23]))
plt.axis('off')
# ----------------------
# /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:30: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
# ----------------------
# (20000, 200, 60, 3)
# (20000, 7)
# (18000, 200, 60, 3) (2000, 200, 60, 3)
# ----------------------
# (-0.5, 59.5, 199.5, -0.5)
# ----------------------
# Let's create a new CTCLayer by subclassing
class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        #label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        labels_mask = 1 - tf.cast(tf.equal(y_true, -1), dtype="int64")
        labels_length = tf.reduce_sum(labels_mask, axis=1)
        loss = self.loss_fn(y_true, y_pred, input_length, tf.expand_dims(labels_length, -1))
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

def build_model():
    # Inputs to the model
    input_img = layers.Input(shape=(200,60,3), name="image", dtype="float32") 
    labels = layers.Input(name="label", shape=(7, ), dtype="float32")

    # First conv block
    x = layers.Conv2D(32,(3, 3),activation="relu",kernel_initializer="he_normal",padding="same",name="Conv1")(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(64,(3, 3),activation="relu",kernel_initializer="he_normal",padding="same",name="Conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model 
    x = layers.Reshape(target_shape=(50, 960), name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = layers.Dense(len(characters)+1, activation="softmax", name="dense2")(x) # 20 = 19 characters + UKN

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="ocr_cnn_lstm_model")

    # Compile the model and return
    model.compile(optimizer=keras.optimizers.Adam())
    return model


# Get the model
model = build_model()
model.summary()
# ----------------------
# Model: "ocr_cnn_lstm_model"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# image (InputLayer)              [(None, 200, 60, 3)] 0
# __________________________________________________________________________________________________
# Conv1 (Conv2D)                  (None, 200, 60, 32)  896         image[0][0]
# __________________________________________________________________________________________________
# pool1 (MaxPooling2D)            (None, 100, 30, 32)  0           Conv1[0][0]
# __________________________________________________________________________________________________
# Conv2 (Conv2D)                  (None, 100, 30, 64)  18496       pool1[0][0]
# __________________________________________________________________________________________________
# pool2 (MaxPooling2D)            (None, 50, 15, 64)   0           Conv2[0][0]
# __________________________________________________________________________________________________
# reshape (Reshape)               (None, 50, 960)      0           pool2[0][0]
# __________________________________________________________________________________________________
# dense1 (Dense)                  (None, 50, 64)       61504       reshape[0][0]
# __________________________________________________________________________________________________
# dropout (Dropout)               (None, 50, 64)       0           dense1[0][0]
# __________________________________________________________________________________________________
# bidirectional (Bidirectional)   (None, 50, 256)      197632      dropout[0][0]
# __________________________________________________________________________________________________
# bidirectional_1 (Bidirectional) (None, 50, 128)      164352      bidirectional[0][0]
# __________________________________________________________________________________________________
# label (InputLayer)              [(None, 7)]          0
# __________________________________________________________________________________________________
# dense2 (Dense)                  (None, 50, 21)       2709        bidirectional_1[0][0]
# __________________________________________________________________________________________________
# ctc_loss (CTCLayer)             (None, 50, 21)       0           label[0][0]
#                                                                  dense2[0][0]
# ==================================================================================================
# Total params: 445,589
# Trainable params: 445,589
# Non-trainable params: 0
# __________________________________________________________________________________________________
# ----------------------
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.map(lambda x,y: {'image':x, 'label':y}).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
print(train_dataset)


validation_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
validation_dataset = validation_dataset.map(lambda x,y: {'image':x, 'label':y}).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
# ----------------------
# <PrefetchDataset shapes: {image: (None, 200, 60, 3), label: (None, 7)}, types: {image: tf.float32, label: tf.int32}>
# ----------------------
# Add early stopping
# early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
# EPOCH
# Train the model
checkpoint_path = "training_1/{epoch:02d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 verbose=1,
                                                 monitor='val_accuracy',
                                                 mode='max',
                                                 save_weights_only=True,
                                                 save_freq='epoch')

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCH,
    batch_size=BATCH_SIZE,
    callbacks=[cp_callback],
    verbose=1
)
os.listdir(checkpoint_dir)
latest = tf.train.latest_checkpoint(checkpoint_path)

# import pickle

# with open('/trainHistoryDict', 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)
# with open('/trainHistoryDict', "rb") as file_pi:
#     history = pickle.load(file_pi)
# ----------------------
# Epoch 1/6
# 1125/1125 [==============================] - 337s 291ms/step - loss: 18.3466 - val_loss: 15.9734

# Epoch 00001: saving model to training_1/01.ckpt
# Epoch 2/6
# 1125/1125 [==============================] - 322s 286ms/step - loss: 6.7949 - val_loss: 1.4619

# Epoch 00002: saving model to training_1/02.ckpt
# Epoch 3/6
# 1125/1125 [==============================] - 317s 282ms/step - loss: 1.5322 - val_loss: 0.7357

# Epoch 00003: saving model to training_1/03.ckpt
# Epoch 4/6
# 1125/1125 [==============================] - 321s 285ms/step - loss: 0.9483 - val_loss: 0.6499

# Epoch 00004: saving model to training_1/04.ckpt
# Epoch 5/6
# 1125/1125 [==============================] - 316s 280ms/step - loss: 0.7752 - val_loss: 0.4850

# Epoch 00005: saving model to training_1/05.ckpt
# Epoch 6/6
# 1125/1125 [==============================] - 313s 278ms/step - loss: 0.6257 - val_loss: 0.4927

# Epoch 00006: saving model to training_1/06.ckpt
# ----------------------
np.save('my_history.npy',history.history)
FileLink(r'my_history.npy')
history=np.load('my_history.npy',allow_pickle='TRUE').item()

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('CTC loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()
# ----------------------
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
prediction_model.summary()
prediction_model.save("model.h5")

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("weights.h5")
FileLink(r'weights.h5')

print("Saved model to disk")

FileLink(r'model.json')

def get_model(path):
    model = keras.models.load_model(path)
    return model
# ----------------------
# Model: "model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# image (InputLayer)           [(None, 200, 60, 3)]      0
# _________________________________________________________________
# Conv1 (Conv2D)               (None, 200, 60, 32)       896
# _________________________________________________________________
# pool1 (MaxPooling2D)         (None, 100, 30, 32)       0
# _________________________________________________________________
# Conv2 (Conv2D)               (None, 100, 30, 64)       18496
# _________________________________________________________________
# pool2 (MaxPooling2D)         (None, 50, 15, 64)        0
# _________________________________________________________________
# reshape (Reshape)            (None, 50, 960)           0
# _________________________________________________________________
# dense1 (Dense)               (None, 50, 64)            61504
# _________________________________________________________________
# dropout (Dropout)            (None, 50, 64)            0
# _________________________________________________________________
# bidirectional (Bidirectional (None, 50, 256)           197632
# _________________________________________________________________
# bidirectional_1 (Bidirection (None, 50, 128)           164352
# _________________________________________________________________
# dense2 (Dense)               (None, 50, 21)            2709
# =================================================================
# Total params: 445,589
# Trainable params: 445,589
# Non-trainable params: 0
# _________________________________________________________________
# Saved model to disk
# ----------------------
m = get_model("model.h5")
y_pred = m.predict(X_val)
y_pred = keras.backend.ctc_decode(y_pred, input_length=np.ones(X_val.shape[0])*50, greedy=True) # decoding -> y_pred[0].shape = (104,5)
#y_pred[0][0][X][0:7] corresponds to the prediction of one image (with X in [0,...,103])
#it is a tensor whose corresponding numpy array is for example [15,  7, 15, 12,  8]
y_pred = y_pred[0][0][0:X_val.shape[0],0:7].numpy() 
#array([[ 2,  2,  8,  0,  0],
#       [15,  7, 15, 12,  8],
#       ...
#       [13, 17, 13, 16,  2],
#       [10,  3, 14,  4,  4]])
# ----------------------
FileLink(r'model.h5')
# ----------------------
# model.h5
# ----------------------
nrow = 1
fig=plt.figure(figsize=(20, 5))
for i in range(0,10):
    if i>4: nrow = 2
    fig.add_subplot(nrow, 5, i+1)
    plt.imshow(X_val[i].transpose((1,0,2)),cmap='gray')
    pred_txt = ''.join(list(map(lambda x:num_to_char[str(x)] if x>-1 else '', y_pred[i])))
    plt.title('Prediction : ' + pred_txt)
    plt.axis('off')
plt.show()
# ----------------------
compute_perf_metric(y_pred, y_val)
# ----------------------
# 0.9770714285714286
# ----------------------
def create_test_dataset():
    X, y = [],[]
    for item in test_dataset:
#         /kaggle/input/russian-captcha-images-base64/translit/images
        img = tf.io.read_file(f"../input/russian-captcha-images-base64/translit/images/{item[1]}")
        img = tf.io.decode_jpeg(img, channels=3) 
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.transpose(img, perm=[1, 0, 2])
        img = img.numpy()
        X.append(img)
        y.append(item[0])
    # y = tf.keras.preprocessing.sequence.pad_sequences(y, 7, padding='post', value=-1)
    X = np.asarray(X)
    y = np.asarray(y)

    return X,y

X_test,file_names = create_test_dataset()
# ----------------------
test_pred = m.predict(X_test)
test_pred = keras.backend.ctc_decode(test_pred, input_length=np.ones(X_test.shape[0])*50, greedy=True)
test_pred = test_pred[0][0][0:X_test.shape[0],0:7].numpy()
# ----------------------
answers = ["".join(list(map(lambda x:num_to_char[str(x)], label))).replace("UKN",'') for label in test_pred]
answers[:10]
# ----------------------
# ['6ж7ж5',
#  '6ж7рк',
#  '6ж824',
#  '6ж895',
#  '6ж96л',
#  '6ж9рр',
#  '6ж9с9',
#  '6ж9тр',
#  '6жбмм',
#  '6жбт4']
# ----------------------

# ----------------------

# ----------------------

# ----------------------

# ----------------------

# ----------------------

# ----------------------

# ----------------------

# ----------------------

# ----------------------

# ----------------------

# ----------------------

# ----------------------

# ----------------------
