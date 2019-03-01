import tensorflow as tf

# z = wx + b
if __name__ == '__main__':
    # Create graph
    g = tf.Graph()
    with g.as_default():
        x = tf.placeholder(dtype=tf.float32, shape=(None), name='x')
        w = tf.Variable(2.0, name='weight')
        b = tf.Variable(0.7, name='bias')
        z = w * x + b

        init = tf.global_variables_initializer()

    #  Create a session and pass in graph g
    with tf.Session(graph=g) as sess:
        # initialize w and b
        sess.run(init)

        # evaluate z
        for input in [1.0, 0.6, -1.8]:
            print('x=%4.1f, z=%4.1f ' % (input, sess.run(z, feed_dict={x: input})))
