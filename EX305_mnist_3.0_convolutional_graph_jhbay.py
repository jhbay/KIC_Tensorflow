# encoding: UTF-8
# original source : https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/tree/master/tensorflow-mnist-tutorial
# 2018.12 : modified by Seungkwon Lee(kahnlee@naver.com)

import tensorflow as tf
# import tensorflowvisu
import math
import mnistdata
print("Tensorflow version " + tf.__version__)
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from statistics import *

from datetime import datetime

def get_logdir(exerciseName) :
	now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
	root_logdir = './cnn_logs'
	logdir = "{}/run-{}/".format(root_logdir, exerciseName + now)
	return logdir

def getStats(dataList, modelName, listSize):
    minList = []
    maxList = []
    meanList = []
    idxList = []
    for i in range(0, len(dataList), listSize): # start, stop, step
        partList = dataList[i:i + listSize]
        print('index::{} ~ {} // mean: {}'.format(i, i + listSize, mean(partList) ) )
        minList.append(min(partList))
        maxList.append(max(partList))
        meanList.append(mean(partList))
        idxList.append(i + listSize)
    statResult = pd.DataFrame({modelName+'_min':minList, modelName+'_max':maxList, modelName+'_mean':meanList }, index = idxList)
    # print('statResult::', statResult)
    return statResult

def getStatSet(dataDict, fileName, listSize):
    statSetResult = pd.DataFrame()
    for modelName in dataDict.keys():
        print('dataDict[modelName]:{}, modelName:{}, listSize:{}'.format(dataDict[modelName][:5], modelName, listSize))
        statSetResult = pd.concat([statSetResult, getStats(dataDict[modelName], modelName, listSize)], axis=1)  #[statSetResult, list(map(float, getStats(dataDict[modelName], modelName, listSize)))]

    print('statSetResult::', statSetResult)
    statSetResult.to_csv(fileName + "_Result.csv", mode='w')



tf.set_random_seed(0)

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnistdata.read_data_sets("data", one_hot=True, reshape=False)

# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                 X [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 5x5x1=>4 stride 1        W1 [5, 5, 1, 4]        B1 [4]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                           Y1 [batch, 28, 28, 4]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x4=>8 stride 2        W2 [5, 5, 4, 8]        B2 [8]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                             Y2 [batch, 14, 14, 8]
#     @ @ @ @ @ @       -- conv. layer 4x4x8=>12 stride 2       W3 [4, 4, 8, 12]       B3 [12]
#     ∶∶∶∶∶∶∶∶∶∶∶                                               Y3 [batch, 7, 7, 12] => reshaped to YY [batch, 7*7*12]
#      \x/x\x\x/        -- fully connected layer (relu)         W4 [7*7*12, 200]       B4 [200]
#       · · · ·                                                 Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)      W5 [200, 10]           B5 [10]
#        · · ·                                                  Y [batch, 10]

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# step for variable learning rate
step = tf.placeholder(tf.int32)

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 4  # first convolutional layer output depth
L = 8  # second convolutional layer output depth
M = 12  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.ones([K])/10)
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/10)
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.ones([M])/10)

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N])/10)
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.ones([10])/10)

# The model
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, the learning rate is a placeholder
# the learning rate is: # 0.0001 + 0.003 * (1/e)^(step/2000)), i.e. exponential decay from 0.003->0.0001
lr = 0.0001 +  tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)


#------------- tensorboard
# Create a summary to monitor cost & accuracy
tf.summary.scalar("loss", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()
# create tensorboard writer object
log_folder = get_logdir('EX305_CNN_')
summary_writer = tf.summary.FileWriter(log_folder, graph=tf.get_default_graph())

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


train_acc_list = []
test_acc_list = []
train_loss_list = []
test_loss_list = []

totalRun = 100;
splitCntBy = 25;

# run
for i in range(totalRun + 1) :

	batch_X, batch_Y = mnist.train.next_batch(100)
	a, c = sess.run([accuracy, cross_entropy],  feed_dict={X: batch_X, Y_: batch_Y, step: i})
	
	if(i%splitCntBy == 0):
		print("training : ", i, ' accuracy = ', '{:7.4f}'.format(a), ' loss = ', c)
	train_acc_list.append(a)
	train_loss_list.append(c)

	# write tensorboard log
	if i % 10 == 0 :
		summary = sess.run(merged_summary_op, feed_dict={X: batch_X, Y_: batch_Y})
		summary_writer.add_summary(summary, i)

	# test_batch_X, test_batch_Y = mnist.test.next_batch(100)  ==> never use mini batch!!
	# sess.run(train_step, feed_dict={X: test_batch_X, Y_: test_batch_Y})  ==> never run train_step on test data!!
	a, c = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
	if(i%splitCntBy == 0):
		print("testing* : ",i, ' accuracy = ', '{:7.4f}'.format(a), ' loss = ', c)
	test_acc_list.append(a)
	test_loss_list.append(c)

	# the backpropagation training step
	sess.run(train_step, {X: batch_X, Y_: batch_Y, step: i})

# print('train_acc_list::', train_acc_list)
# print('test_acc_list::', test_acc_list)
# print('train_loss_list::', train_loss_list)
# print('test_loss_list::', test_loss_list)

dataDict = {'train_acc':train_acc_list,'test_acc':test_acc_list,'train_loss':train_loss_list,'test_loss':test_loss_list}
getStatSet(dataDict, "../python_DL/cnn_logs/time", splitCntBy)


# draw graph : accuracy
x = np.arange(len(train_acc_list))
plt.figure(1) 
plt.plot(x, train_acc_list,  label='train', markevery=1)
plt.plot(x, test_acc_list, label='test', markevery=1)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
# plt.show()

# draw graph : loss
x = np.arange(len(train_loss_list))
plt.figure(2) 
plt.plot(x, train_loss_list,  label='train', markevery=1)
plt.plot(x, test_loss_list, label='test', markevery=1)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.ylim(0, 100)
plt.legend(loc='upper right')
plt.show()