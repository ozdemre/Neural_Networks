import tensorflow as tf

# Create TensorFlow object called hello_constant
hello_constant = tf.constant('Hello World!')
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)
math_op_add=tf.add(3,5)
print(math_op_add)
#tf.subtract(tf.constant(2.0),tf.constant(1)) #throw an error lets cast to data type
math_op_substract=(tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1)))

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    #output = sess.run(hello_constant)
    #output = sess.run(x, feed_dict={x: 'Hello World'})
    #output = sess.run(z, feed_dict={x: 'Hello World', y: 123, z: 45.67})
    output = sess.run(math_op_add)
    print(output)

