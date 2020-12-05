# Solution is available in the other "solution.py" tab
import numpy as np


def softmax(z):
    """Compute softmax values for each sets of scores in z."""
    # TODO: Compute and return softmax(z)
    return np.exp(z) / np.sum(np.exp(z), axis=0)

logits = [3.0, 1.0, 0.2]
print(softmax(logits))
# logits is a two-dimensional array
logits = np.array([
    [1, 2, 3, 6],
    [2, 4, 5, 6],
    [3, 8, 7, 6]])
# softmax will return a two-dimensional array with the same shape
print(softmax(logits))