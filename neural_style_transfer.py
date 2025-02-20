import tensorflow as tf
import numpy as np
from keras.applications import vgg19  #pretrained VGG19 model
from keras.preprocessing.image import load_img,img_to_array
# from keras.api.models import Model
from io import BytesIO
from PIL import Image 

def load_preprocess_img(img_path):
    if isinstance(img_path,Image.Image):
        img = img_path.resize((400,400))
    else:
        img = load_img(img_path,target_size=(400,400)) #load img and convert t0 400x400
    img = img_to_array(img) 
    img = np.expand_dims(img,axis=0) # After adding batch dimension: (400, 400, 3) --> (1, 400, 400, 3)
    img = vgg19.preprocess_input(img) #Normalize image for vgg19
    return img


def image_to_bytes(img):
    img_byte_arr = BytesIO()
    img.save(img_byte_arr,format="JPEG")
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def deprocess_img(img):
    img = img.reshape((400,400,3)).astype('float32') #convert to 400x400x3
    img[:,:,0] += 103.939 #red channel
    img[:,:,1] += 116.779 #green channel
    img[:,:,2] += 123.68 #blue channel
    img = np.clip(img,0,255).astype('uint8') #ensure pixel  intensity values are between 0 and 255
    return img

content_layer = 'block5_conv2' #deep layer that captures content features
style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1'] #shallow and deep layers that capture different artistic details

# def get_model(saved_path = "vgg19_model"):
#     vgg = vgg19.VGG19(weights='imagenet',include_top=False) #we have removed the final fully connected layers
#     vgg.trainable = False # freeze the VGG19 weights
#     outputs = {layer.name : layer.output for layer in vgg.layers}
#     model = Model(inputs=vgg.input,outputs=outputs)
#     model.save(saved_path)
#     print(f"Saved model at {saved_path}")



def compute_content_loss(content,generated):
    return np.mean(np.square(content-generated)) #mena squared error

def gram_matrix(tensor):
    tensor = tf.squeeze(tensor,axis=0) # Remove batch dimension (1, H, W, C) → (H, W, C)
    channels = int(tensor.shape[-1])
    x = tf.reshape(tf.transpose(tensor,(2,0,1)),[channels,-1]) # (C, H*W)
    return tf.matmul(x,tf.transpose(x))/(tensor.shape[0]*tensor.shape[1]*channels)


def compute_style_loss(style,generated):
    return tf.reduce_mean(tf.square(gram_matrix(style)-gram_matrix(generated)))

def compute_total_loss(model,content_img,style_img,generated_img,alpha=1.0,beta=1e3):
    content_features = model(content_img)[content_layer]
    generated_content_features = model(generated_img)[content_layer]
    content_loss = compute_content_loss(content_features,generated_content_features)

    style_loss = 0
    for layer in style_layers:
        style_features = model(style_img)[layer]
        generated_style_features = model(generated_img)[layer]

        style_loss += compute_style_loss(style_features,generated_style_features)

    return alpha*content_loss + beta*style_loss


