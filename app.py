import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/dev/null"

import streamlit as st
from PIL import Image
#from neural_style_transfer import load_preprocess_img,get_model,tf,compute_total_loss,deprocess_img,image_to_bytes
# -*- coding: utf-8 -*-
"""neural_style_transfer.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SBH3D3hZHe-E_yeuApY7CAtLpLIJH4El
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.api.applications import vgg19  #pretrained VGG19 model
from keras.api.preprocessing.image import load_img,img_to_array
from keras.api.models import Model
import keras.api.backend as K
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
vgg = vgg19.VGG19(weights='imagenet',include_top=False)
#we have removed the final fully connected layers


content_layer = 'block5_conv2' #deep layer that captures content features
style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1'] #shallow and deep layers that capture different artistic details

def get_model():
    vgg.trainable = False # freeze the VGG19 weights
    outputs = {layer.name : layer.output for layer in vgg.layers}
    model = Model(inputs=vgg.input,outputs=outputs)
    return model

model = get_model()

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


def main():
    st.title("Neural Style Transfer")
    content_img_file = st.file_uploader("Upload Content Image",type=['jpg','jpeg','png'])
    style_img_file = st.file_uploader("Upload Style Image",type=['jpg','jpeg','png'])

    if content_img_file is not None and style_img_file is not None:
        content_img = Image.open(content_img_file).resize((400,400))
        style_img = Image.open(style_img_file).resize((400,400))

        st.image(content_img,caption="Content Image",use_container_width=True)
        st.image(style_img,caption="Style Image",use_container_width=True)

        content_img_preprocessed = load_preprocess_img(content_img)
        style_img_preprocessed = load_preprocess_img(style_img)

        model = get_model()
        #content_layer = 'block5_conv2'
        #style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']

        generated_img = tf.Variable(content_img_preprocessed,dtype=tf.float32)
        optimizer = tf.optimizers.Adam(learning_rate=5.0)

        epochs = 500
        for i in range(epochs):
          with tf.GradientTape() as tape:
            loss = compute_total_loss(model, content_img_preprocessed, style_img_preprocessed,generated_img)
          grad = tape.gradient(loss, generated_img)  # Compute gradients
          optimizer.apply_gradients([(grad, generated_img)])  # Update image

          if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.numpy()}")

        final_img = deprocess_img(generated_img.numpy())
        st.image(final_img,caption="Generated Image",use_column_width=True)
        st.download_button(label="Download Generated Image",data=image_to_bytes(final_img),file_name="generated_image.jpg",mime="image/jpeg")

if __name__ == "__main__":
    main()


