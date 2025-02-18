import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg19  #pretrained VGG19 model 
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def load_preprocess_img(img_path):
    img = load_img(img_path,target_size=(400,400)) #load img and convert t0 400x400
    img = img_to_array(img) 
    img = np.expand_dims(img,axis=0) # After adding batch dimension: (400, 400, 3) --> (1, 400, 400, 3)
    img = vgg19.preprocess_input(img) #Normalize image for vgg19
    return img
img = load_preprocess_img(r"C:\Users\HP\OneDrive\Desktop\ML\NEURAL_STYLE_TRANSFER\mycat.jpg")
print(img.shape)

def deprocess_img(img):
    img = img.reshape((400,400,3))
    img[:,:,0] += 103.939
    img[:,:,1] += 116.779
    img[:,:,2] += 123.68
    img = np.clip(img,0,255).astype('uint8')
    return img
img = deprocess_img(img)
print(img.shape)
content_img_path = r"C:\Users\HP\OneDrive\Desktop\ML\NEURAL_STYLE_TRANSFER\mycat.jpg"
style_img_path = r"C:\Users\HP\OneDrive\Desktop\ML\NEURAL_STYLE_TRANSFER\style.jpg"

content_img = load_preprocess_img(content_img_path)
style_img = load_preprocess_img(style_img_path)

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
    return K.mean(K.square(content-generated)) #mean squared error

def gram_matrix(tensor):
    tensor = K.squeeze(tensor,axis=0) # Remove batch dimension (1, H, W, C) → (H, W, C)
    channels = int(tensor.shape[-1])
    x = K.batch_flatten(K.permute_dimensions(tensor,(2,0,1))) # (C, H*W)
    return K.dot(x,K.transpose(x))/(tensor.shape[0]*tensor.shape[1]*channels)

def compute_style_loss(style,generated):
    return K.mean(K.square(gram_matrix(style)-gram_matrix(generated)))

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

generated_img = tf.Variable(content_img,dtype=tf.float32)
optimizer = tf.optimizers.Adam(learning_rate=5.0)

epochs = 500
for i in range(epochs):
    with tf.GradientTape() as tape:
        loss = compute_total_loss(model, content_img, style_img, generated_img)
    grad = tape.gradient(loss, generated_img)  # Compute gradients
    optimizer.apply_gradients([(grad, generated_img)])  # Update image

    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss.numpy()}")
