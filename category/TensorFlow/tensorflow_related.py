import tensorflow as tf
import time

flags = tf.flags
FLAGS = flags.FLAGS

# # Required parameters # #
# # flags.DEFINE_string("paras_name", "default_value", "illustration")
flags.DEFINE_string(
    "data_dir", "./data/",
    "The input data dir. Should contain the .tsv files (or other data files).")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")


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

    # tf.clip_by_value()
    a = tf.constant([0, 0, 1.0001, 0.0001])
    y = tf.clip_by_value(a, 1e-2, 1)
    g = tf.gradients(y, a, stop_gradients=a)
    sess.run(init)
    print(sess.run(y))
    print(sess.run(g))


def main(_):
    start_time = time.time()

    # # --------- paras received from terminal-------- #
    print(FLAGS.data_dir)
    print(FLAGS.do_train)
    print(FLAGS.learning_rate)

    # # ------------ whether_differentiable----------- #
    # whether_differentiable()
    # print('Operation takes {}s'.format(time.time()-start_time))


if __name__ == '__main__':
    # # pop a UserWarning for necessary parameters
    # # (paras that has a None default value)
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("do_train")
    flags.mark_flag_as_required("learning_rate")
    tf.app.run()
