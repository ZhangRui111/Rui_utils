import tensorflow as tf
import time


def whether_differentiable():
    """
    Use tf.gradient to find out whether an operation is derivable in Tensorflow (< 1.8.0).
    If there are some operations nondifferentiable in y, error appears.
    :return:
    """
    sess = tf.Session()
    init = tf.global_variables_initializer()

    # tf.gradients
    a = tf.constant(2.)
    b = 2 * a
    y = a + b
    g = tf.gradients(y, [a, b], stop_gradients=[a, b])
    sess.run(init)
    print(sess.run(y))
    print(sess.run(g))

    # Above is equivalent to:
    a = tf.stop_gradient(tf.constant(2.))
    b = tf.stop_gradient(2 * a)
    y = a + b
    g = tf.gradients(y, [a, b])
    sess.run(init)
    print(sess.run(y))
    print(sess.run(g))

    # Attention stop_gradients=[a] rather than [a, b]
    a = tf.constant(2.)
    b = 2 * a
    y = a + b
    g = tf.gradients(y, [a, b], stop_gradients=[a])
    sess.run(init)
    print(sess.run(y))
    print(sess.run(g))

    # tf.math.ceil()
    a = tf.constant(1.8)
    b = tf.ceil(-1.8)
    y = a * b
    g = tf.gradients(y, [a, b], stop_gradients=[a, b])
    sess.run(init)
    print(sess.run(y))
    print(sess.run(g))

    # tf.one_hot()
    a = tf.constant([1, 2, 3, 4])
    b = tf.one_hot(2, 4, on_value=2, off_value=1)
    y = a * b
    g = tf.gradients(y, [a, b], stop_gradients=[a, b])
    sess.run(init)
    print(sess.run(y))
    print(sess.run(g))


def main():
    start_time = time.time()

    whether_differentiable()

    print('Operation takes {}s'.format(time.time()-start_time))


if __name__ == '__main__':
    main()
