import os
import numpy as np
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2

from tensorflow.keras.layers import Conv2D,Activation,BatchNormalization
from tensorflow.keras.layers import UpSampling2D,Input,Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras.metrics import Recall,Precision
from tensorflow.keras import backend as K

np.random.seed(42)
tf.random.set_seed(42)

IMAGE_SIZE=256
EPOCHS=1000
BATCH=8
LR=1e-4
PATH="C:\\Projects\\Foot Ulcer Segmentation Challenge"
model_save_path="C:\\Projects\\Foot Ulcer Segmentation Challenge\\Checkpoints"

def read_image(path):
    path=path.decode()
    x=cv2.imread(path,cv2.IMREAD_COLOR)
    x=cv2.resize(x,(IMAGE_SIZE,IMAGE_SIZE))
    x=x/255.0
    return x

def read_mask(path):
    path=path.decode()
    y=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    y=cv2.resize(y,(IMAGE_SIZE,IMAGE_SIZE))
    y=y/255.0
    y=np.expand_dims(y,axis=-1)
    return y

def read_and_rgb(x):
    x=cv2.imread(x)
    x=cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
    return x

def tf_parse(x,y):
    def _parse(x,y):
        x=read_image(x)
        y=read_mask(y)
        return x,y
    x,y=tf.numpy_function(_parse,[x,y],[tf.float64,tf.float64])
    x.set_shape([IMAGE_SIZE,IMAGE_SIZE,3])
    y.set_shape([IMAGE_SIZE,IMAGE_SIZE,1])
    return x,y

def tf_dataset(x,y,batch=8):
    dataset=tf.data.Dataset.from_tensor_slices((x,y))
    dataset=dataset.map(tf_parse)
    dataset=dataset.batch(batch)
    dataset=dataset.repeat()
    return dataset


def load_data(path):
    
    train_x=sorted(glob(os.path.join(path,"train/images/*")))
    train_y=sorted(glob(os.path.join(path,"train/labels/*")))
    valid_x=sorted(glob(os.path.join(path,"validation/images/*")))
    valid_y=sorted(glob(os.path.join(path,"validation/labels/*")))
    
    return (train_x,train_y),(valid_x,valid_y)                
    
(train_x,train_y),(valid_x,valid_y)=load_data(PATH)

fig = plt.figure(figsize=(15, 15))
a = fig.add_subplot(1, 4, 1)
imgplot = plt.imshow(read_and_rgb(train_x[0]))

a = fig.add_subplot(1, 4, 2)
imgplot = plt.imshow(read_and_rgb(train_x[1]))
imgplot.set_clim(0.0, 0.7)

a = fig.add_subplot(1, 4, 3)
imgplot = plt.imshow(read_and_rgb(train_x[2]))
imgplot.set_clim(0.0, 1.4)

a = fig.add_subplot(1, 4, 4)
imgplot = plt.imshow(read_and_rgb(train_x[3]))
imgplot.set_clim(0.0, 2.1)

fig = plt.figure(figsize=(15, 15))
a = fig.add_subplot(1, 4, 1)
imgplot = plt.imshow(read_and_rgb(train_y[0]))

a = fig.add_subplot(1, 4, 2)
imgplot = plt.imshow(read_and_rgb(train_y[1]))
imgplot.set_clim(0.0, 0.7)

a = fig.add_subplot(1, 4, 3)
imgplot = plt.imshow(read_and_rgb(train_y[2]))
imgplot.set_clim(0.0, 1.4)

a = fig.add_subplot(1, 4, 4)
imgplot = plt.imshow(read_and_rgb(train_y[3]))
imgplot.set_clim(0.0, 1.4)

def model():
    inputs=Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3),name="input_image")
    encoder=MobileNetV2(input_tensor=inputs,weights="imagenet",include_top=False,alpha=0.35)
    skip_connection_names=["input_image","block_1_expand_relu","block_3_expand_relu","block_6_expand_relu"]
    encoder_output=encoder.get_layer("block_13_expand_relu").output
    f=[16,32,48,64]
    x=encoder_output
    print("Encoder Output Shape",x.shape)
    print(encoder.summary())
    #Decoder Beginning
    for i in range(1,len(skip_connection_names)+1,1):
        x_skip=encoder.get_layer(skip_connection_names[-i]).output
        x=UpSampling2D((2,2))(x)
        x=Concatenate()([x,x_skip])
        
        x=Conv2D(f[-i],(3,3),padding="same")(x)
        x=BatchNormalization()(x)
        x=Activation("relu")(x)
        
        x=Conv2D(f[-i],(3,3),padding="same")(x)
        x=BatchNormalization()(x)
        x=Activation("relu")(x)
        
    x=Conv2D(1,(1,1),padding="same")(x)
    x=Activation("sigmoid")(x)
    model=Model(inputs,x)
    return model

smooth=1e-15
def dice_coef(y_true,y_pred):
    y_true=tf.keras.layers.Flatten()(y_true)
    y_pred=tf.keras.layers.Flatten()(y_pred)
    intersection=tf.reduce_sum(y_true*y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

train_dataset = tf_dataset(train_x, train_y, batch=BATCH)
valid_dataset = tf_dataset(valid_x, valid_y, batch=BATCH)
checkpt=ModelCheckpoint(filepath=model_save_path,monitor='loss',save_best_only=True,verbose=1,mode='min')
model=model()
opt=tf.keras.optimizers.Nadam(LR)
metrics=[dice_coef,Recall(),Precision()]
model.compile(loss=dice_loss,optimizer=opt,metrics=metrics)
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False),
    checkpt
]

train_steps = len(train_x)//BATCH
valid_steps = len(valid_x)//BATCH

if len(train_x) % BATCH != 0:
    train_steps += 1
if len(valid_x) % BATCH != 0:
    valid_steps += 1

model.fit(train_dataset,validation_data=valid_dataset,epochs=EPOCHS,steps_per_epoch=train_steps,
         validation_steps=valid_steps,callbacks=callbacks)
