import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

def convert_to_functional(seq_model, input_shape=(224, 224, 3)):
    tf.keras.backend.clear_session()
    
    input_layer = Input(shape=input_shape, name="input_node")
    x = input_layer
    
    for i, layer in enumerate(seq_model.layers):
        # We get the layer's config but remove the name to let Keras 
        # generate a fresh, unique one based on the original type
        config = layer.get_config()
        if "name" in config:
            config["name"] = f"func_{config['name']}_{i}"
        
        # Recreate the layer from config and apply it to our tensor x
        new_layer = type(layer).from_config(config)
        x = new_layer(x)
    
    func_model = Model(inputs=input_layer, outputs=x)
    
    # Manually transfer weights
    for i, layer in enumerate(seq_model.layers):
        func_model.layers[i+1].set_weights(layer.get_weights())
        
    return func_model