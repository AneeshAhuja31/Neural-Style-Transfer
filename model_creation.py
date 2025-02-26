from keras.api.applications import vgg19  #pretrained VGG19 model
from keras.api.models import Model
import tensorflow as tf
def get_model(saved_path = "vgg19_model.keras"):
    tf.keras.backend.clear_session()
    vgg = vgg19.VGG19(weights='imagenet',include_top=False) #we have removed the final fully connected layers
    vgg.trainable = False # freeze the VGG19 weights
    outputs = {layer.name : layer.output for layer in vgg.layers}
    model = Model(inputs=vgg.input,outputs=outputs)
    return model

if __name__ == '__main__':
    get_model()