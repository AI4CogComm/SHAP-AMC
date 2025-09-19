import pandas as pd
import matplotlib as plt
import tensorflow as tf
import numpy as np
import rml128_dtiny
import os
from imp import reload
from keras.layers import Input,Dense,Conv1D,MaxPool1D,ReLU,BatchNormalization,Dropout,Softmax,Attention
from keras.layers import LSTM,CuDNNLSTM,Bidirectional,Flatten,Reshape,Concatenate,Layer,GlobalMaxPooling1D
from keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
import cleverhans
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method
import carlini_wagner_l2

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

reload(rml128_dtiny)
(mods,snrs,lbl),(X_train1,Y_train1),(X_val1,Y_val1),(X_test1,Y_test1),(train_idx1,val_idx1,test_idx1) = \
    rml128_dtiny.load_data()
classes = mods

(mods,snrs,lbl),(X_train2,Y_train2),(X_val2,Y_val2),(X_test2,Y_test2),(train_idx2,val_idx2,test_idx2) = \
    rml128_dtiny.load_data()

(mods,snrs,lbl),(X_train3,Y_train3),(X_val3,Y_val3),(X_test3,Y_test3),(train_idx3,val_idx3,test_idx3) = \
    rml128_dtiny.load_data()

(mods,snrs,lbl),(X_train4,Y_train4),(X_val4,Y_val4),(X_test4,Y_test4),(train_idx4,val_idx4,test_idx4) = \
    rml128_dtiny.load_data()

model = get_model(11)
model.compile(optimizer=Adam(learning_rate = 0.001), loss='categorical_crossentropy',metrics=['accuracy'])
model = tf.keras.models.load_model('model')
loss1, accuracy1 = model.evaluate(x=X_test1, y=Y_test1)
print("第一个测试集的准确率为: {:.2f}%".format(accuracy1 * 100))
print("第一个测试集的损失为: {:.2f}%".format(loss1 * 100))

loss_object = tf.keras.losses.CategoricalCrossentropy()
def create_adversarial_pattern(input_signal, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_signal)
        prediction = model(input_signal)
        loss = loss_object(input_label, prediction)
    gradient = tape.gradient(loss, input_signal)
    # 对梯度使用sign函数，创建扰动
    signed_grad = tf.sign(gradient)
    return signed_grad

def perturbation_gen(X,Y,eps):
    perturbations = [] #用于存储生成的干扰，最终和信号的形状一样
    for i in range(len(X)):
        x_i =  tf.convert_to_tensor(np.reshape(X[i].copy(),(1,128,2)))
        y_i =  tf.convert_to_tensor(np.reshape(Y[i].copy(),(1,11)))
        perturbations.append(create_adversarial_pattern(x_i, y_i))#*0.5+0.5可以转换到0-1
    perturbations = np.squeeze(np.array(perturbations))#生成最终的干扰信号(6600,128,2)
    for i in range(len(X)):
        range_am_i = np.max(X[i,:,0])-np.min(X[i,:,0])
        range_ph_i = np.max(X[i,:,1])-np.min(X[i,:,1])
        perturbations[i,:,0] = range_am_i * perturbations[i,:,0]
        perturbations[i,:,1] = range_ph_i * perturbations[i,:,1]
    return X.copy() + np.squeeze(eps*perturbations)

#pgd
eps = 0.025
num_iter = 10
eps_iterr = 0.001
rand_init = 0.3*eps
xtrain_adv1 = projected_gradient_descent(model, X_train1.copy(), eps, eps_iterr,num_iter, np.inf,
                                         clip_min=-1,clip_max=1,rand_init=rand_init, rand_minmax=0.3)
loss, accuracy = model.evaluate(x=xtrain_adv1, y=Y_train1)
print("第1个训练集的准确率为: {:.2f}%".format(accuracy * 100))
print("第1个训练集的损失为: {:.2f}%".format(loss * 100))
xval_adv1 = projected_gradient_descent(model, X_val1.copy(), eps, eps_iterr,num_iter, np.inf,
                                       clip_min=-1,clip_max=1,rand_init=rand_init, rand_minmax=0.3)
loss, accuracy = model.evaluate(x=xval_adv1, y=Y_val1)
print("第1个验证集的准确率为: {:.2f}%".format(accuracy * 100))
print("第1个验证集的损失为: {:.2f}%".format(loss * 100))

#fgsm
xtrain_adv2 = perturbation_gen(X_train2.copy(),Y_train2.copy(),eps)
loss, accuracy = model.evaluate(x=xtrain_adv2, y=Y_train2)
print("第2个训练集的准确率为: {:.2f}%".format(accuracy * 100))
print("第2个训练集的损失为: {:.2f}%".format(loss * 100))
xval_adv2 = perturbation_gen(X_val2.copy(),Y_val2.copy(),eps)
loss, accuracy = model.evaluate(x=xval_adv2, y=Y_val2)
print("第2个验证集的准确率为: {:.2f}%".format(accuracy * 100))
print("第2个验证集的损失为: {:.2f}%".format(loss * 100))

#bim
eps_iter = (eps/0.025)*0.001
xtrain_adv3 =  basic_iterative_method(model, X_train3.copy(), eps=eps, eps_iter=eps_iter ,nb_iter=num_iter, norm=np.inf,
                                      # clip_min=-1,clip_max=1,
                                      rand_init=None,rand_minmax=0.3)
loss, accuracy = model.evaluate(x=xtrain_adv3, y=Y_train3)
print("第3个训练集的准确率为: {:.2f}%".format(accuracy * 100))
print("第3个训练集的损失为: {:.2f}%".format(loss * 100))
xval_adv3 = basic_iterative_method(model, X_val3.copy(), eps=eps, eps_iter=eps_iter ,nb_iter=num_iter, norm=np.inf,
                                      # clip_min=-1,clip_max=1,
                                   rand_init=None,rand_minmax=0.3)
loss, accuracy = model.evaluate(x=xval_adv3, y=Y_val3)
print("第3个验证集的准确率为: {:.2f}%".format(accuracy * 100))
print("第3个验证集的损失为: {:.2f}%".format(loss * 100))

max_i=5
xtrain_adv4 = carlini_wagner_l2.carlini_wagner_l2(model,X_train4.copy(),binary_search_steps=5,max_iterations=max_i,
                                                confidence=0.0,initial_const=0.5,learning_rate=0.03,)
loss, accuracy = model.evaluate(x=xtrain_adv4, y=Y_train4)
print("第4个训练集的准确率为: {:.2f}%".format(accuracy * 100))
print("第4个训练集的损失为: {:.2f}%".format(loss * 100))
xval_adv4 = carlini_wagner_l2.carlini_wagner_l2(model,X_val4.copy(),binary_search_steps=5,max_iterations=max_i,
                                                confidence=0.0,initial_const=0.5,learning_rate=0.03,)
loss, accuracy = model.evaluate(x=xval_adv4, y=Y_val4)
print("第4个验证集的准确率为: {:.2f}%".format(accuracy * 100))
print("第4个验证集的损失为: {:.2f}%".format(loss * 100))

xtrainl = np.concatenate((xtrain_adv1,xtrain_adv2,xtrain_adv3,xtrain_adv4,X_train1,X_train2,X_train3,X_train4),axis=0)
xvall = np.concatenate((xval_adv1,xval_adv2,xval_adv3,xval_adv4,X_val1,X_val2,X_val3,X_val4),axis=0)
print(xtrainl.shape,xvall.shape)
ytrainl = np.zeros((len(xtrainl)))
inter = int(len(ytrainl)/2)
ytrainl[:inter]=1
yvall = np.zeros((len(xvall)))
inter1 = int(len(yvall)/2)
yvall[:inter1]=1
print(xtrainl.shape)
print(xvall.shape)
print(ytrainl.shape)
print(yvall.shape)

#定义攻击检测模型并训练
class SumLayer(Layer):
    def __init__(self, **kwargs):
        super(SumLayer, self).__init__(**kwargs)

    def call(self, inputs):

        summed_output = tf.reduce_sum(inputs, axis=1)

        return summed_output
def detect_model(n_outputs=1):
    inputs = Input(shape=(128, 2))
    conv1 = Conv1D(filters=128, kernel_size=8, activation='relu', kernel_initializer="glorot_uniform")(inputs)
    lstm1 = LSTM(128, return_sequences=True)(conv1)
    attention = Attention()([lstm1, lstm1])
    sumlayer = SumLayer()(attention)
    BN1 = BatchNormalization()(sumlayer)
    dense1 = Dense(256, activation='relu')(BN1)
    output = Dense(n_outputs, activation='sigmoid')(dense1)  # 输出一个概率值
    model = Model(inputs=inputs, outputs=output)
    return model
detect_model = detect_model(1)
detect_model.summary()
filepath = 'detect_model'
detect_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
history = detect_model.fit(xtrainl, ytrainl, epochs=200, verbose=1,batch_size=64,validation_data=(xvall,yvall),
                     callbacks=[
                         keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                         keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, verbose=1, patience=10,min_lr=0.000001),
                         keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto')
                     ]
                     )
detect_model =  tf.keras.models.load_model('detect_model')

#检测模型性能
reload(rml128_dtiny)
(mods,snrs,lbl),(X_train11,Y_train11),(X_val11,Y_val11),(X_test11,Y_test11),(train_idx11,val_idx11,test_idx11) = \
    rml128_dtiny.load_data()
classes = mods

eps = 0.025

xtest_adv11 = projected_gradient_descent(model, X_test11.copy(), eps, eps_iterr,num_iter, np.inf,
                                       clip_min=-1,clip_max=1,rand_init=rand_init, rand_minmax=0.3)
# xtest_adv11 = perturbation_gen(X_test11.copy(),Y_test11.copy(), eps)

# eps_iter = (eps/0.025)*0.001
# xtest_adv11 = basic_iterative_method(model, X_test11.copy(), eps, eps_iter=eps_iter ,nb_iter=num_iter, norm=np.inf,
#                                       rand_init=None,rand_minmax=0.3)

# xtest_adv11 = carlini_wagner_l2.carlini_wagner_l2(model,X_test11.copy(),binary_search_steps=5,max_iterations=10,
#                                                 confidence=0.0,initial_const=0.5,learning_rate=0.03,)

ytest11 = np.ones((len(xtest_adv11)))
loss, accuracy = detect_model.evaluate(x=xtest_adv11, y=ytest11)
print("准确率为: {:.2f}%".format(accuracy * 100))

ytest11 = np.zeros((len(X_test11)))
loss, accuracy = detect_model.evaluate(x=X_test11, y=ytest11)
print("准确率为: {:.2f}%".format(accuracy * 100))













