import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot

import tempfile

input_shape = [20]
x_train = np.random.randn(1, 20).astype(np.float32)
y_train = tf.keras.utils.to_categorical(np.random.randn(1), num_classes=20)

def setup_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(20, input_shape=input_shape),
      tf.keras.layers.Flatten()
  ])
  return model

def setup_pretrained_weights():
  model= setup_model()

  model.compile(
      loss=tf.keras.losses.categorical_crossentropy,
      optimizer='adam',
      metrics=['accuracy']
  )

  model.fit(x_train, y_train)

  _, pretrained_weights = tempfile.mkstemp('.tf')

  model.save_weights(pretrained_weights)

  return pretrained_weights

def setup_pretrained_model():
  model = setup_model()
  pretrained_weights = setup_pretrained_weights()
  model.load_weights(pretrained_weights)
  return model

# Create a base model
base_model = setup_pretrained_model()

# 通过这个指定想要qat量化的layer
def apply_quantization_to_dense(layer):
  if isinstance(layer, tf.keras.layers.Dense):
    return tfmot.quantization.keras.quantize_annotate_layer(layer)
  return layer

annotated_model = tf.keras.models.clone_model(
    base_model,
    clone_function=apply_quantization_to_dense,
)

# Now that the Dense layers are annotated,
# `quantize_apply` actually makes the model quantization aware.
quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
quant_aware_model.summary()
