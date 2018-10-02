import tensorflow as tf

# define two different graph with different operations
"""
a = tf.get_variable("a1", initializer=tf.constant(3))
b = tf.get_variable("b1", initializer=tf.constant(8))
c = tf.multiply(a, b)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(c)
    print (sess.run(c))
"""
g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v", shape=[1], initializer=tf.zeros_initializer())
    a = tf.get_variable("a1", initializer=tf.constant(30))
    b = tf.get_variable("b1", initializer=tf.constant(80))
    c = tf.add(a, b, name="c1")

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v", shape=[1], initializer=tf.ones_initializer())
    a = tf.get_variable("a1", initializer=tf.constant(30))
    a = tf.get_variable("a2", initializer=tf.constant(3))
    b = tf.get_variable("b2", initializer=tf.constant(8))
    c = a * b

with g1.as_default():
    with tf.Session(graph=g1) as sess:
        sess.run(tf.global_variables_initializer())
        with tf.variable_scope("", reuse=True):
            print (sess.run(v.eval()))

with tf.Session(graph=g2) as sess:
    sess.run(tf.global_variables_initializer())
    with tf.variable_scope("", reuse=True):
        print (sess.run(tf.get_variable("v")))



writer = tf.summary.FileWriter('./log/3_1', tf.get_default_graph())
writer.close()
