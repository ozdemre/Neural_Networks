import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import pickle
import math
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
import matplotlib.pyplot as plt
from tqdm import tqdm

#%matplotlib inline

print('All modules imported.')

# Load the data
pickle_file = 'notMNIST_random_sample_150000.pickle'
with open(pickle_file, 'rb') as f:
  pickle_data = pickle.load(f)
  train_features = pickle_data['train_features']
  train_labels = pickle_data['train_labels']
  test_features = pickle_data['test_features']
  test_labels = pickle_data['test_labels']

  # Set flags for feature engineering.  This will prevent you from skipping an important step.
  is_features_normal = False
  is_labels_encod = False
  del pickle_data  # Free up memory

print('Data loaded.')


def normalize_grayscale(image_data):
  """
  Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
  :param image_data: The image data to be normalized
  :return: Normalized image data
  """
  # TODO: Implement Min-Max scaling for grayscale image data
  a = 0.1
  b = 0.9
  grayscale_min = 0
  grayscale_max = 255
  return a + (((image_data - grayscale_min) * (b - a)) / (grayscale_max - grayscale_min))


### DON'T MODIFY ANYTHING BELOW ###
# Test Cases
np.testing.assert_array_almost_equal(
  normalize_grayscale(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255])),
  [0.1, 0.103137254902, 0.106274509804, 0.109411764706, 0.112549019608, 0.11568627451, 0.118823529412, 0.121960784314,
   0.125098039216, 0.128235294118, 0.13137254902, 0.9],
  decimal=3)
np.testing.assert_array_almost_equal(
  normalize_grayscale(np.array([0, 1, 10, 20, 30, 40, 233, 244, 254, 255])),
  [0.1, 0.103137254902, 0.13137254902, 0.162745098039, 0.194117647059, 0.225490196078, 0.830980392157, 0.865490196078,
   0.896862745098, 0.9])

if not is_features_normal:
  train_features = normalize_grayscale(train_features)
  test_features = normalize_grayscale(test_features)
  is_features_normal = True

print('Tests Passed!')

if not is_labels_encod:
    # Turn labels into numbers and apply One-Hot Encoding
    encoder = LabelBinarizer()
    encoder.fit(train_labels)
    train_labels = encoder.transform(train_labels)
    test_labels = encoder.transform(test_labels)

    # Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)
    is_labels_encod = True

print('Labels One-Hot Encoded')

assert is_features_normal, 'You skipped the step to normalize the features'
assert is_labels_encod, 'You skipped the step to One-Hot Encode the labels'

# Get randomized datasets for training and validation
train_features, valid_features, train_labels, valid_labels = train_test_split(
    train_features,
    train_labels,
    test_size=0.05,
    random_state=832289)

print('Training features and labels randomized and split.')

features_count = 784
labels_count = 10

# Problem 2 - Set the features and labels tensors
features = tf.placeholder(tf.float32)
labels = tf.placeholder(tf.float32)

# Problem 2 - Set the weights and biases tensors
weights = tf.Variable(tf.truncated_normal((features_count, labels_count)))
biases = tf.Variable(tf.zeros(labels_count))


### DON'T MODIFY ANYTHING BELOW ###

#Test Cases
from tensorflow.python.ops.variables import Variable

assert features._op.name.startswith('Placeholder'), 'features must be a placeholder'
assert labels._op.name.startswith('Placeholder'), 'labels must be a placeholder'
assert isinstance(weights, Variable), 'weights must be a TensorFlow variable'
assert isinstance(biases, Variable), 'biases must be a TensorFlow variable'

assert features._shape == None or (\
    features._shape.dims[0].value is None and\
    features._shape.dims[1].value in [None, 784]), 'The shape of features is incorrect'
assert labels._shape  == None or (\
    labels._shape.dims[0].value is None and\
    labels._shape.dims[1].value in [None, 10]), 'The shape of labels is incorrect'
assert weights._variable._shape == (784, 10), 'The shape of weights is incorrect'
assert biases._variable._shape == (10), 'The shape of biases is incorrect'

assert features._dtype == tf.float32, 'features must be type float32'
assert labels._dtype == tf.float32, 'labels must be type float32'

# Feed dicts for training, validation, and test session
train_feed_dict = {features: train_features, labels: train_labels}
valid_feed_dict = {features: valid_features, labels: valid_labels}
test_feed_dict = {features: test_features, labels: test_labels}

# Linear Function WX + b
logits = tf.matmul(features, weights) + biases

prediction = tf.nn.softmax(logits)

# Cross entropy
cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), axis=1)

# Training loss
loss = tf.reduce_mean(cross_entropy)

# Create an operation that initializes all variables
init = tf.global_variables_initializer()

# Test Cases
with tf.Session() as session:
    session.run(init)
    session.run(loss, feed_dict=train_feed_dict)
    session.run(loss, feed_dict=valid_feed_dict)
    session.run(loss, feed_dict=test_feed_dict)
    biases_data = session.run(biases)

assert not np.count_nonzero(biases_data), 'biases must be zeros'

print('Tests Passed!')

# Determine if the predictions are correct
is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
# Calculate the accuracy of the predictions
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

print('Accuracy function created.')

# TODO: Set the epochs, batch_size, and learning_rate with the best parameters from problem 3
#Parameter configurations:
###Configuration 1
#Epochs: 1
#Batch Size: 2000 1000 500 300 50
#Learning Rate: 0.01
###Configuration 2
#Epochs: 1
#Batch Size: 100
#Learning Rate: 0.8 0.5 0.1 0.05 0.01
###Configuration 3
#Epochs: 1 2 3 4 5
#Batch Size: 100
#Learning Rate: 0.2
epochs = 5
batch_size = 100
learning_rate = 0.1

### DON'T MODIFY ANYTHING BELOW ###
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# The accuracy measured against the validation set
validation_accuracy = 0.0

# Measurements use for graphing loss and accuracy
log_batch_step = 50
batches = []
loss_batch = []
train_acc_batch = []
valid_acc_batch = []

with tf.Session() as session:
  session.run(init)
  batch_count = int(math.ceil(len(train_features) / batch_size))

  for epoch_i in range(epochs):

    # Progress bar
    batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i + 1, epochs), unit='batches')

    # The training cycle
    for batch_i in batches_pbar:
      # Get a batch of training features and labels
      batch_start = batch_i * batch_size
      batch_features = train_features[batch_start:batch_start + batch_size]
      batch_labels = train_labels[batch_start:batch_start + batch_size]

      # Run optimizer and get loss
      _, l = session.run(
        [optimizer, loss],
        feed_dict={features: batch_features, labels: batch_labels})

      # Log every 50 batches
      if not batch_i % log_batch_step:
        # Calculate Training and Validation accuracy
        training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
        validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

        # Log batches
        previous_batch = batches[-1] if batches else 0
        batches.append(log_batch_step + previous_batch)
        loss_batch.append(l)
        train_acc_batch.append(training_accuracy)
        valid_acc_batch.append(validation_accuracy)

    # Check accuracy against Validation data
    validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

loss_plot = plt.subplot(211)
loss_plot.set_title('Loss')
loss_plot.plot(batches, loss_batch, 'g')
loss_plot.set_xlim([batches[0], batches[-1]])
acc_plot = plt.subplot(212)
acc_plot.set_title('Accuracy')
acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
acc_plot.plot(batches, valid_acc_batch, 'x', label='Validation Accuracy')
acc_plot.set_ylim([0, 1.0])
acc_plot.set_xlim([batches[0], batches[-1]])
acc_plot.legend(loc=4)
plt.tight_layout()
plt.show()

print('Validation accuracy at {}'.format(validation_accuracy))

### DON'T MODIFY ANYTHING BELOW ###
# The accuracy measured against the test set
test_accuracy = 0.0

with tf.Session() as session:
  session.run(init)
  batch_count = int(math.ceil(len(train_features) / batch_size))

  for epoch_i in range(epochs):

    # Progress bar
    batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i + 1, epochs), unit='batches')

    # The training cycle
    for batch_i in batches_pbar:
      # Get a batch of training features and labels
      batch_start = batch_i * batch_size
      batch_features = train_features[batch_start:batch_start + batch_size]
      batch_labels = train_labels[batch_start:batch_start + batch_size]

      # Run optimizer
      _ = session.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})

    # Check accuracy against Test data
    test_accuracy = session.run(accuracy, feed_dict=test_feed_dict)

assert test_accuracy >= 0.80, 'Test accuracy at {}, should be equal to or greater than 0.80'.format(test_accuracy)
print('Nice Job! Test Accuracy is {}'.format(test_accuracy))