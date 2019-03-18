import numpy as np
import tensorflow.keras as keras

from data import SaveMNISTData as mnist_data


def create_batch_generator(X, y, batch_size=128, shuffle=False):
    X_copy = np.array(X)
    y_copy = np.array(y)

    if shuffle:
        data = np.column_stack((X_copy, y_copy))
        np.random.shuffle(data)
        X_copy = data[:, :-1]
        y_copy = data[:, -1].astype(int)

    for i in range(0, X.shape[0], batch_size):
        yield (X_copy[i:i + batch_size, :], y_copy[i:i + batch_size])


if __name__ == '__main__':
    X_train, y_train = mnist_data.load('../data/', kind='train')
    print('Rows: %d,  Columns: %d' % (X_train.shape[0], X_train.shape[1]))

    X_test, y_test = mnist_data.load('../data/', kind='t10k')
    print('Rows: %d,  Columns: %d' % (X_test.shape[0], X_test.shape[1]))

    mean_vals = np.mean(X_train, axis=0)
    std_val = np.std(X_train)

    X_train_centered = (X_train - mean_vals) / std_val
    X_test_centered = (X_test - mean_vals) / std_val

    del X_train, X_test

    print(X_train_centered.shape, y_train.shape)
    print(X_test_centered.shape, y_test.shape)

    n_features = X_train_centered.shape[1]
    n_classes = 10
    random_seed = 123
    np.random.seed(random_seed)

    y_train_onehot = keras.utils.to_categorical(y_train)
    print('First 3 labels: ', y_train[:3])

    model = keras.models.Sequential()

    model.add(
        keras.layers.Dense(
            units=50,
            input_dim=X_train_centered.shape[1],
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='tanh'))

    model.add(
        keras.layers.Dense(
            units=50,
            input_dim=50,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='tanh'))

    model.add(
        keras.layers.Dense(
            units=y_train_onehot.shape[1],
            input_dim=50,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='softmax'))

    sgd_optimizer = keras.optimizers.SGD(lr=0.001, decay=1e-7, momentum=.9)

    model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy')

    history = model.fit(X_train_centered, y_train_onehot,
                        batch_size=64, epochs=50,
                        verbose=1,
                        validation_split=0.1)

    y_train_pred = model.predict_classes(X_train_centered, verbose=0)
    print('First 3 predictions: ', y_train_pred[:3])

    correct_preds = np.sum(y_train == y_train_pred, axis=0)
    train_acc = correct_preds / y_train.shape[0]
    print('Training accuracy: %.2f%%' % (train_acc * 100))

    y_test_pred = model.predict_classes(X_test_centered,verbose=0)
    correct_preds = np.sum(y_test == y_test_pred, axis=0)
    test_acc = correct_preds / y_test.shape[0]
    print('Test accuracy: %.2f%%' % (test_acc * 100))
