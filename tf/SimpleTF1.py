import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    g = tf.Graph()
    with g.as_default():
        x = tf.placeholder(dtype='float', shape=(None, 2, 3), name='input_x')
        x2 = tf.reshape(x, shape=(-1, 6), name='x2')
        xsum = tf.reduce_sum(x2, axis=0, name='col_sum')
        xmean = tf.reduce_mean(x2, axis=0, name='cols_mean')

    with tf.Session(graph=g) as sess:
        x_array = np.arange(24).reshape(4, 2, 3)
        print('Input: /n', x_array)
        print('Input shape: ', x_array.shape)
        print('Reshaped: /n', sess.run(x2, feed_dict={x: x_array}))
        print('Sum: /n', sess.run(xsum, feed_dict={x: x_array}))
        print('Mean: /n', sess.run(xmean, feed_dict={x: x_array}))
