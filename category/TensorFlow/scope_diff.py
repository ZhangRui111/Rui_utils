"""
It is for the purpose of the variable sharing mechanism that a separate type
of scope (variable scope) was introduced.
As a result, we end up having two different types of scopes:

- name scope, created using tf.name_scope
- variable scope, created using tf.variable_scope

Both scopes have the same effect on all operations as well as variables created
using tf.Variable, i.e., the scope will be added as a prefix to the operation or
variable name.
However, name scope is ignored by tf.get_variable.
"""

import tensorflow as tf

with tf.name_scope("foo"):
    with tf.variable_scope("var_scope"):
        with tf.name_scope("foo2"):
            a = tf.get_variable(name="a", shape=[1])
            b = tf.Variable(1, name="b", dtype=tf.float32)
with tf.name_scope("bar"):
    with tf.variable_scope("var_scope", reuse=True):
        with tf.name_scope("bar2"):
            a1 = tf.get_variable("a", shape=[1])
            b1 = tf.Variable(1, name="b", dtype=tf.float32)

print(a.name)   # var_scope/var:0
print(a1.name)   # var_scope/var:0
print(id(a.name))
print(id(a1.name))
assert a == a1
assert id(a) == id(a1)
print(b.name)  # foo/var_scope/foo2/w:0
print(b1.name)  # bar/var_scope/bar2/w:0
print(id(b))
print(id(b1))
try:
    assert id(b) == id(b1)
except AssertionError:
    print("id(b) != id(b1)")

try:
    with tf.name_scope("bar"):
        with tf.variable_scope("var_scope"):
            with tf.name_scope("foo2"):
                a = tf.get_variable("a", shape=[1])
except ValueError as err:
    print(err)

try:
    with tf.name_scope("bar"):
        with tf.variable_scope("var_scope", reuse=True):
            with tf.name_scope("foo2"):
                a2 = tf.get_variable("a", shape=[2])
except ValueError as err:
    print(err)
