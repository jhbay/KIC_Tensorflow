# encoding: UTF-8
# 2018.12 : created by Seungkwon Lee(kahnlee@naver.com)

import tensorflow as tf

x_data = [1, 2, 3, 4]
y_data = [2, 4, 6, 8]
 
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W')
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='b')

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# hypothesis = W * X + b
hypothesis = tf.add(tf.multiply(W, X), b, name='hypothesis')

cost = tf.reduce_mean(tf.square(hypothesis - Y), name='cost')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost, name='train_op')

with tf.Session() as sess:
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())

	for step in range(1, 2001) :
		_, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

		print(step, cost_val, sess.run(W), sess.run(b))

		# save model
		if step % 100 == 0 :
			saver.save(sess, './model/my_model', global_step=step)

	print("\n=== Test ===")
	print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
	print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: 2.5}))
