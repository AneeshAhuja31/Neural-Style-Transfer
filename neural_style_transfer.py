import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"  # Prevent memory allocation errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings

import tensorflow as tf
import numpy as np
from cv2 import (
    cvtColor, calcHist, LUT, COLOR_RGB2LAB, COLOR_LAB2RGB,
    split, merge, HISTCMP_CORREL
)
from keras.api.applications import vgg19  #pretrained VGG19 model
from keras.api.preprocessing.image import load_img,img_to_array
# from keras.api.models import Model
from io import BytesIO
from PIL import Image,ImageEnhance


def load_preprocess_img(img_path):
    if isinstance(img_path,Image.Image):
        img = img_path.resize((400,400))
    else:
        img = load_img(img_path,target_size=(400,400)) #load img and convert t0 400x400

     # Convert grayscale to RGB if needed
    if img.mode not in ['RGB', 'RGBA']:
        img = img.convert('RGB')
    
    elif img.mode == 'RGBA':
        img = img.convert('RGB')
    img = img_to_array(img) 
    img = np.expand_dims(img,axis=0) # After adding batch dimension: (400, 400, 3) --> (1, 400, 400, 3)
    img = vgg19.preprocess_input(img) #Normalize image for vgg19
    return img


def image_to_bytes(img,format="JPEG"):
    img_byte_arr = BytesIO()
    img.save(img_byte_arr,format=format)
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def deprocess_img(img,rgb_or_rgba,original_alpha=None):
    img = img.reshape((400,400,3)).astype('float32') #convert to 400x400x3
    img[:,:,0] += 103.939 #red channel
    img[:,:,1] += 116.779 #green channel
    img[:,:,2] += 123.68 #blue channel
    img = np.clip(img,0,255).astype('uint8') #ensure pixel  intensity values are between 0 and 255
    
    img = Image.fromarray(img,"RGB")
    if rgb_or_rgba and original_alpha is not None:
        img = img.convert("RGBA")
        original_alpha_img = Image.fromarray(original_alpha,"L")
        img.putalpha(original_alpha_img)
    return img

content_layer = 'block5_conv2' #deep layer that captures content features
style_layers = ['block1_conv1','block2_conv1','block3_conv1'] #shallow and deep layers that capture different artistic details


def compute_content_loss(content,generated):
    return np.mean(np.square(content-generated)) #mena squared error

def gram_matrix(tensor):
    tensor = tf.squeeze(tensor,axis=0) # Remove batch dimension (1, H, W, C) → (H, W, C)
    channels = int(tensor.shape[-1])
    x = tf.reshape(tf.transpose(tensor,(2,0,1)),[channels,-1]) # (C, H*W)
    return tf.matmul(x,tf.transpose(x))/(tensor.shape[0]*tensor.shape[1]*channels)


def compute_style_loss(style,generated):
    return tf.reduce_mean(tf.square(gram_matrix(style)-gram_matrix(generated)))

#used to ensure that the generated image is visually smooth
def compute_total_variation_loss(img):
    x_variation = img[:,:,1:,:] - img[:,:,:-1,:]#difference in neighbouring horizontal pixels
    y_variation = img[:,1:,:,:] - img[:,:-1,:,:]#difference in neighbouring vertical pixels
    return tf.reduce_sum(tf.abs(x_variation)) + tf.reduce_sum(tf.abs(y_variation))

def compute_total_loss(model,content_img,style_img,generated_img,alpha=0.5,beta=2e3,gamma=30):
    content_features = model(content_img)[content_layer]
    generated_content_features = model(generated_img)[content_layer]
    content_loss = compute_content_loss(content_features,generated_content_features)

    style_loss = 0
    for layer in style_layers:
        style_features = model(style_img)[layer]
        generated_style_features = model(generated_img)[layer]
        style_loss += compute_style_loss(style_features,generated_style_features)
    tv_loss = compute_total_variation_loss(generated_img)
    return alpha*content_loss + beta*style_loss + gamma*tv_loss

#enhance contrast after generation
def enhance_contrast(img_array,factor=1.5):
    img = Image.fromarray(np.uint8(img_array))
    enhancer = ImageEnhance.Contrast(img)
    enhanced_img = enhancer.enhance(factor)
    return np.array(enhanced_img)

def match_histograms(source, reference):
    # """
    # Matches the color histogram of the source image to the reference image.

    # Args:
    #     source (numpy.ndarray or PIL.Image.Image): The source image.
    #     reference (numpy.ndarray or PIL.Image.Image): The reference image.

    # Returns:
    #     numpy.ndarray: The source image with the color histogram matched to the reference.
    # """

    # Convert to numpy arrays if necessary
    # Convert to numpy arrays if necessary
    if isinstance(source, Image.Image):
        source = np.array(source)
    if isinstance(reference, Image.Image):
        reference = np.array(reference)

    # Convert to float32
    src_float = np.clip(source, 0, 255).astype(np.uint8)
    ref_float = np.clip(reference, 0, 255).astype(np.uint8)

    # Convert to LAB color space
    src_lab = cvtColor(src_float, COLOR_RGB2LAB)
    ref_lab = cvtColor(ref_float, COLOR_RGB2LAB)

    # Split into channels
    src_l, src_a, src_b = split(src_lab)
    ref_l, ref_a, ref_b = split(ref_lab)

    # Match histograms for each channel
    matched_l = match_channel_histogram(src_l, ref_l)
    matched_a = match_channel_histogram(src_a, ref_a)
    matched_b = match_channel_histogram(src_b, ref_b)

    # Merge channels back
    matched_lab = merge([matched_l, matched_a, matched_b])
    matched_lab = matched_lab.astype(np.uint8)

    # Convert back to RGB
    matched_rgb = cvtColor(matched_lab, COLOR_LAB2RGB)

    return np.clip(matched_rgb, 0, 255).astype(np.uint8)


def match_channel_histogram(source, reference):
    # """
    # Matches the histogram of a single channel of the source image to the reference image.

    # Args:
    #     source (numpy.ndarray): The source image channel (single channel).
    #     reference (numpy.ndarray): The reference image channel (single channel).

    # Returns:
    #     numpy.ndarray: The source image channel with the histogram matched to the reference.
    # """
    # Compute histograms
    source_hist = calcHist([source], [0], None, [256], [0, 256])
    reference_hist = calcHist([reference], [0], None, [256], [0, 256])

    # Compute cumulative distribution function
    source_cdf = source_hist.cumsum()
    reference_cdf = reference_hist.cumsum()

    # Normalize CDFs
    source_cdf = source_cdf / source_cdf[-1]
    reference_cdf = reference_cdf / reference_cdf[-1]

    # Create mapping function
    lookup_table = np.interp(source_cdf.flatten(), reference_cdf.flatten(), np.arange(256))

    # Apply mapping to source image
    return LUT(source, lookup_table.astype(np.uint8))