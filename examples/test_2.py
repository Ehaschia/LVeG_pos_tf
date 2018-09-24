import numpy as np
import tensorflow as tf

# state_0 = np.array([0.79338586, 0])
# state_1 = np.array([0.5076815, 0])
# state_2 = np.array([1.0465684, 0])
#
# trans_0 = np.array([[0, 1.4772854], [1.1929934, 0]])
# trans_1 = np.array([[0, 1.456225], [0.6111443, 0]])
#
# op1 = state_0.reshape([2, 1]) + trans_0
# op2 = np.log(np.sum(np.exp(op1), axis=0))
# op3 = state_1 + op2
# op4 = op3.reshape([2, 1]) + trans_1
# op5 = np.log(np.sum(np.exp(op4), axis=0))
# op6 = state_2 + op5
#
# res = np.log(np.sum(np.exp(op6), axis=0))
# print(res)

np.random.seed(1)
tf.set_random_seed(1)
BATCH = 2
num_tag = 3
dim = 3
idx0 = np.array([0, 1])
idx1 = np.array([0, 1])
# here batch is 2
labels = np.random.rand(num_tag, num_tag, dim)

tf_lables = tf.placeholder(tf.float32, shape=(num_tag, num_tag, dim), name='lables')
tf_idx0 = tf.placeholder(tf.int32, shape=(BATCH), name='lables')
tf_idx1 = tf.placeholder(tf.int32, shape=(BATCH), name='lables')

label0 = tf.gather(tf_lables, tf_idx0)
flattened_label0 = tf.reshape(label0, [-1, dim])
flattened_idx = tf.range(BATCH)*num_tag + tf_idx1
flattened_label1 = tf.gather(flattened_label0, flattened_idx)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
feed_dict = {tf_idx0:idx0,
             tf_idx1:idx1,
             tf_lables:labels}
res = sess.run(flattened_label1, feed_dict=feed_dict)
print(res)
print('='*10)
print(labels[1, 1])