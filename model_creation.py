from keras.api.applications import vgg19  #pretrained VGG19 model
from keras.api.models import Model
from keras.api.saving import save_model


def get_model(saved_path = "vgg19_model.keras"):
    vgg = vgg19.VGG19(weights='imagenet',include_top=False) #we have removed the final fully connected layers
    vgg.trainable = False # freeze the VGG19 weights
    outputs = {layer.name : layer.output for layer in vgg.layers}
    model = Model(inputs=vgg.input,outputs=outputs)
    save_model(model,saved_path)
    print(f"Saved model at {saved_path}")

if __name__ == '__main__':
    get_model()