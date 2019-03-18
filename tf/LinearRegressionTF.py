import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def build_graph(g):
    with g.as_default():
        tf.set_random_seed(999)

        tf_x = tf.placeholder(shape=None, dtype=tf.float32, name='tf_x')
        tf_y = tf.placeholder(shape=None, dtype=tf.float32, name='tf_y')

        weight = tf.Variable(tf.random_normal(shape=(1, 1), stddev=0.25), name='weight')
        bias = tf.Variable(0.0, name='bias')

        y_hat = tf.add(weight * tf_x, bias, name='y_hat')

        cost = tf.reduce_mean(tf.square(tf_y - y_hat), name='cost')
        optim = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optim.minimize(cost, name='train_op')
    return tf_x, tf_y, cost, train_op, g


def make_random_data():
    x = np.random.uniform(low=-1, high=4, size=200)
    y = []
    for t in x:
        r = np.random.normal(loc=0.0, scale=(0.5 + t * t / 3), size=None)
        y.append(r)
    return x, 1.726 * x - 0.84 + np.array(y)


def train_model(X, y_hit, cost, train_op, graph):
    training_costs = []
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(n_epochs):
            c, _ = sess.run([cost, train_op], feed_dict={X: x_train, y_hit: y_train})
            training_costs.append(c)

            if not e % 50:
                print('Epoch %4d: %.4f' % (e, c))
    return training_costs


if __name__ == '__main__':
    g = tf.Graph()
    x_raw, y_raw = make_random_data()
    plt.plot(x_raw, y_raw, 'o')
    plt.show()

    x_train, y_train = x_raw[:100], y_raw[:100]
    x_test, y_test = x_raw[100:], y_raw[100:]

    n_epochs = 500

    X, y_hit, cost, train_op, graph = build_graph(g)
    training_costs = train_model(X, y_hit, cost, train_op, graph)

    plt.plot(training_costs)
    plt.show()
