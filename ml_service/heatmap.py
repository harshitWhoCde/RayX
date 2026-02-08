import tensorflow as tf
import numpy as np

def generate_heatmap(img_array, model, last_conv_layer_name):
    # We create a sub-model to extract the conv activations and the final output
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        # Cast input to float32 to ensure precision
        img_array = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(img_array)
        
        # Get the index of the highest predicted class
        class_channel = predictions[:, tf.argmax(predictions[0])]

    # Gradients of the top class with respect to the last conv layer
    grads = tape.gradient(class_channel, conv_outputs)

    # Average the gradients across the width and height
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map by "how important it is"
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU to only keep features that have a POSITIVE influence on the class
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    
    return heatmap.numpy()