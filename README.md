# This Repo Will Help You To Get Started With TensorFlow


### Tensor Creation and Attributes

* **`tf.constant()`**: Creates a tensor whose value cannot be changed. This is used for fixed values in your model.
* **`tf.Variable()`**: Creates a tensor whose value can be changed. This is useful for variables that are updated during model training, like weights.
* **`tf.ones()`**: Creates a tensor of a specified shape and fills it with ones.
* **`tf.zeros()`**: Creates a tensor of a specified shape and fills it with zeros.
* **`tf.random.uniform()`**: Creates a tensor with random values between 0 and 1, with a specified shape.
* **`tf.random.set_seed()`**: Sets the global random seed for reproducibility.
* **`tensor.shape`**: This attribute returns the dimensions of the tensor, for example, `(2, 3, 4, 5)`.
* **`tensor.ndim`**: Returns the number of dimensions, or rank, of the tensor.
* **`tf.size(tensor)`**: Returns the total number of elements in the tensor.
* **`tensor.dtype`**: Returns the data type of the tensor's elements, such as `tf.float32`.

### Tensor Manipulation

* **`tf.cast()`**: Changes a tensor's data type, for example, from `int32` to `float16` or `float32`.
* **`tf.abs()`**: Computes the absolute value of each element in a tensor.
* **`tf.square()`**: Squares each element in a tensor.
* **`tf.sqrt()`**: Calculates the square root of each element in a tensor. Note that the input tensor must have a float data type.
* **`tf.math.log()`**: Calculates the natural logarithm of each element in a tensor. The input tensor must have a float data type.
* **`tf.add()`, `tf.subtract()`, `tf.multiply()`, `tf.divide()`**: These functions perform element-wise arithmetic operations on tensors.

### Aggregation and Indexing

* **`tf.reduce_min()`**: Finds the minimum value in a tensor.
* **`tf.reduce_max()`**: Finds the maximum value in a tensor.
* **`tf.reduce_mean()`**: Computes the mean of the elements across a tensor.
* **`tf.reduce_sum()`**: Computes the sum of the elements in a tensor.
* **`tf.argmax()`**: Returns the index of the maximum value in a tensor.
* **`tf.argmin()`**: Returns the index of the minimum value in a tensor.
* **`tf.concat()`**: Joins tensors along an existing axis without adding a new dimension.
* **`tf.stack()`**: Stacks tensors along a new axis, adding a new dimension.
* **`tf.squeeze()`**: Removes all dimensions of size 1 from the shape of a tensor.
* **`tf.expand_dims()`**: Adds a new dimension of size 1 at a specified axis.
* **`tf.transpose()`**: Flips the axes of a matrix.
* **`tf.matmul()` or `@` operator**: Performs matrix multiplication.

### Other Important Functions

* **`tf.convert_to_tensor()`**: Converts a NumPy array to a TensorFlow tensor. This is useful when integrating NumPy and TensorFlow.
* **`tf.random.shuffle()`**: Shuffles the order of elements along the first dimension of a tensor. This is useful for shuffling datasets before training.
* **`tf.one_hot()`**: Converts integer indices into one-hot encoded vectors, which is a common way to represent categorical data.
* **`tf.config.list_physical_devices()`**: Lists the available physical devices (CPUs, GPUs, TPUs) that TensorFlow can use.