import tensorflow as tf

# Load your Keras model.
model = tf.keras.models.load_model(r'C:\Users\MANIA\Desktop\grad_project\action.h5')

# Create the converter from your model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Add the option to use Select TF Ops and disable tensor list lowering.
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS     # enable TensorFlow ops.
]
# Disable experimental lowering of tensor list ops.
converter._experimental_lower_tensor_list_ops = False

# Optionally, enable optimizations.
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model.
tflite_model = converter.convert()

# Save the TFLite model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
