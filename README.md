# Target of usage
The layer can be useful, when the input data are XYZ coordinates, as in Molecular Dynamics, Robotics, LIDAR, autopilot driving.

# AngleComparison Layer
The `AngleComparison` layer is an analogy for angle calculation using cosine formula using inner product. It is used in the case when we use X, Y, Z coordinates and want to include information between the features, but we don't want to do it explicitly. 

Sure, here's an updated Github description for the `AngleComparison` layer that includes the additional information you requested:
AngleComparison Layer

The AngleComparison layer is a custom layer for comparing two outputs of a 1D CNN layer in TensorFlow and PyTorch. The layer works by passing an input tensor through two different 1D CNN layers with the same filter size and kernel size. The output of each 1D CNN layer is then compared using a dot product computed with respect to the channel dimension. The dot product is normalized by the product of the L2 norms of the two 1D CNN layers. The final output is a tensor that concatenates the angle tensor with the input tensor.

The AngleComparison layer is an analogy for angle calculation using cosine formula using inner product. It is used in the case when we use X, Y, Z coordinates and want to include information between the features, but we don't want to do it explicitly. So we calculate them for the higher level of abstraction.

The `AngleComparison` layer is useful for applications where we want to compare two outputs of a 1D CNN layer and extract a measure of similarity between them. This layer can be used in many different applications, such as natural language processing, speech processing, and time series analysis.

# Example Usage

Here's an example of how to use the AngleComparison layer in TensorFlow:

```
import tensorflow as tf
from angle_comparison_tensorflow import AngleComparison

inputs = tf.keras.Input(shape=(100, 10))
x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same')(inputs)
x = AngleComparison(filters=32, kernel_size=3)(x)
model = tf.keras.Model(inputs=inputs, outputs=x)


```

And here's an example of how to use the AngleComparison layer in PyTorch:
```
import torch
import torch.nn as nn
from angle_comparison_pytorch import AngleComparison

inputs = torch.randn(32, 100)
x = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)(inputs.unsqueeze(1))
x = AngleComparison(filters=32, kernel_size=3)(x)
model = nn.Sequential(nn.Linear(101, 64), nn.ReLU(), nn.Linear(64, 10))
output = model(x)
```

# Installation

You can install the AngleComparison layer by coping code or cloning this repository and importing respective module.

# License

This project is licensed under the MIT License - see the LICENSE file for details.
