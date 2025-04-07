# TensorFlow Fundamentals - README

This README serves as a beginner-friendly summary and guide based on the notebook `tensorflow00_fundamentals.ipynb`. It covers the basics of working with tensors in TensorFlow, including creation, manipulation, and fundamental operations.

## 📦 Setup
Before starting, make sure to install TensorFlow:
```bash
pip install tensorflow
```

---

## 🔢 Tensor Basics
### Creating Tensors
```python
import tensorflow as tf
rank_2_tensor = tf.constant([[10, 7], [7, 10]])
```
- `.shape` gives tensor shape.
- `.numpy()` converts tensor to NumPy array.

### Tensor Slicing
```python
rank_2_tensor[:, -1]  # Last element from each row
```

### Expanding Dimensions
```python
rank_3_tensor = rank_2_tensor[..., tf.newaxis]
tf.expand_dims(rank_2_tensor, axis=0)
```

---

## ✨ Tensor Operations
- Element-wise: `+`, `-`, `*`, `/`
- Tensor methods: `tf.add()`, `tf.multiply()` (GPU-optimized)

```python
tensors = tf.constant([[10, 7], [7, 10]])
tensors + 10
```

---

## 🧮 Matrix Multiplication
Two rules:
- Inner dimensions must match.
- Output shape: outer dimensions.

```python
tf.matmul(tensors, tensors)
tensors @ tensors  # Alternative
```

### Reshaping for Matrix Multiplication
```python
X = tf.random.uniform((3, 2), maxval=10, dtype=tf.int32)
Y = tf.random.uniform((3, 2), maxval=20, dtype=tf.int32)
Y_reshaped = tf.reshape(Y, (2, 3))
tf.matmul(X, Y_reshaped)
```

### Transposing
```python
tf.transpose(X)
tf.matmul(tf.transpose(X), Y)
```

---

## 📌 Dot Product
```python
tf.tensordot(tf.transpose(X), Y, axes=1)
```

---

## 🔁 Type Casting
```python
B = tf.constant([1.4, 1.7])
D = tf.cast(B, dtype=tf.int16)
```

---

## 📉 Aggregating Tensors
```python
E = tf.constant(np.random.randint(0, 100, size=50))
tf.reduce_min(E)
tf.reduce_max(E)
tf.reduce_mean(E)
tf.reduce_sum(E)
```

## 🎲 Variance & Standard Deviation
```python
import tensorflow_probability as tfp
tfp.stats.variance(E)
tf.math.reduce_std(tf.cast(E, dtype=tf.float32))
```

---

## 🔍 Positional Min/Max
```python
F = tf.random.uniform(shape=[50])
tf.argmax(F)  # index of max value
tf.argmin(F)  # index of min value
```

---

## 🚿 Squeeze Tensors
```python
G = tf.constant(tf.random.uniform([50]), shape=[1, 1, 1, 1, 50])
tf.squeeze(G)  # Removes dimensions with size 1
```

---

## ✅ Summary
You’ve just explored:
- Tensor creation and slicing
- Tensor operations and reshaping
- Matrix multiplication and dot products
- Type casting and aggregation
- Finding min/max values and their indices
- Removing single-dimension entries

This knowledge forms the foundation for deeper learning in TensorFlow, especially for neural networks and computer vision tasks.

Happy TensorFlow-ing! 🚀

