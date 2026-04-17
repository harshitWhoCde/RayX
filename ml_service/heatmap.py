import tensorflow as tf
import numpy as np

def generate_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        img_array = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(img_array)
        
        # Get the index of the highest predicted class
        class_channel = predictions[:, tf.argmax(predictions[0])]

    # Gradients of the top class with respect to the last conv layer
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Safeguard: If gradients disconnect, return a blank heatmap instead of crashing
    if grads is None:
        return np.zeros((img_array.shape[1], img_array.shape[2]))

    # Average the gradients across the width and height
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    
    # 🟢 SENIOR DEV FIX: More stable tensor multiplication
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # ReLU to only keep positive features
    heatmap = tf.maximum(heatmap, 0)
    
    # Safe Normalization
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
        
    return heatmap.numpy()