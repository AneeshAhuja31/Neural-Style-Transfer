import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"  # Prevent memory allocation errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings

import tensorflow as tf
import numpy as np
from cv2 import (
    cvtColor, calcHist, LUT, COLOR_RGB2LAB, COLOR_LAB2RGB,
    split, merge, HISTCMP_CORREL, GaussianBlur, detailEnhance
)
from keras.api.applications import vgg19  
from keras.api.preprocessing.image import load_img,img_to_array
from io import BytesIO
from PIL import Image,ImageEnhance, ImageFilter

IMAGE_SIZE = 400

def load_preprocess_img(img_path):
    if isinstance(img_path,Image.Image):
        img = img_path.resize((400,400))
    else:
        img = load_img(img_path,target_size=(400,400)) 

     # Convert grayscale to RGB if needed
    if img.mode not in ['RGB', 'RGBA']:
        img = img.convert('RGB')
    
    elif img.mode == 'RGBA':
        img = img.convert('RGB')
    img = img_to_array(img) 
    img = np.expand_dims(img,axis=0) # After adding batch dimension: (400, 400, 3) --> (1, 400, 400, 3)
    
    # Manual preprocessing like VGG19
    img_array = img.astype(np.float32)
    img_array[:, :, :, 0] -= 103.939  # Subtract mean R
    img_array[:, :, :, 1] -= 116.779  # Subtract mean G
    img_array[:, :, :, 2] -= 123.68   # Subtract mean B
    
    # Note: VGG uses channels in BGR order, but we keep RGB for simplicity
    return img_array


def image_to_bytes(img,format="JPEG",quality=95):
    img_byte_arr = BytesIO()
    img.save(img_byte_arr,format=format,quality=quality)
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def deprocess_img(img,rgb_or_rgba,original_alpha=None):
    img = img.reshape((IMAGE_SIZE,IMAGE_SIZE,3)).astype('float32') #convert to 400x400x3
    img[:,:,0] += 103.939 #red channel
    img[:,:,1] += 116.779 #green channel
    img[:,:,2] += 123.68 #blue channel
    img = np.clip(img,0,255).astype('uint8') #ensure pixel  intensity values are between 0 and 255
    
    img = Image.fromarray(img,"RGB")
    if rgb_or_rgba and original_alpha is not None:
        #resize alpha channel if needed
        if original_alpha.shape[0] != IMAGE_SIZE or original_alpha.shape[1] != IMAGE_SIZE:
            alpha_img = Image.fromarray(original_alpha,'L')
            alpha_img =  alpha_img.resize((IMAGE_SIZE,IMAGE_SIZE),Image.LANCZOS)
            original_alpha = np.array(alpha_img)
        img = img.convert("RGBA")
        original_alpha_img = Image.fromarray(original_alpha,"L")
        img.putalpha(original_alpha_img)
    return img

CONTENT_LAYERS = 'block5_conv2' #deep layer that captures content features
STYLE_LAYERS = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1'] #shallow and deep layers that capture different artistic details
STYLE_WEIGHTS = [2.0, 1.6, 1.2, 0.8, 0.4]

def compute_content_loss(content,generated):
    return tf.reduce_mean(tf.square(content-generated)) #mena squared error

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
    all_layers = [CONTENT_LAYERS]+STYLE_LAYERS
    
    outputs = model(content_img)
    content_features = {layer:outputs[layer] for layer in all_layers if layer in outputs}
    
    outputs = model(style_img)
    style_features = {layer:outputs[layer] for layer in all_layers if layer in outputs}

    outputs = model(generated_img)
    generated_features = {layer:outputs[layer] for layer in all_layers if layer in outputs}

    #content loss:
    content_loss = compute_content_loss(
        content_features[CONTENT_LAYERS],
        generated_features[CONTENT_LAYERS]
    )
    #style loss
    style_loss = 0
    for i,layer in enumerate(STYLE_LAYERS):
        if layer in style_features and layer in generated_features:
            layer_style_loss = compute_style_loss(
                style_features[layer],
                generated_features[layer]
            )
            style_loss+=STYLE_WEIGHTS[i]*layer_style_loss
    
    tv_loss = compute_total_variation_loss(generated_img)
    # Combine losses
    total_loss = alpha * content_loss + beta * style_loss + gamma * tv_loss
    
    return total_loss

# In model_creation.py or wherever your get_model function is defined
def get_model():
    #force eager execution to avoid name_scope issues
    tf.config.run_functions_eagerly(True)
    # Set up a clean session
    tf.keras.backend.clear_session()
    
    # Load VGG19 with clean initialization
    base_model = vgg19.VGG19(weights='imagenet', include_top=False)
    
    # Create output dictionary for each layer
    outputs = {layer.name:layer.output for layer in base_model.layers}    
    # Create and return the model
    return tf.keras.Model(inputs=base_model.input, outputs=outputs)

#enhance contrast after generation
def enhance_contrast(img_array,factor=1.7):
    img = Image.fromarray(np.uint8(img_array))
    img = ImageEnhance.Contrast(img).enhance(factor)
    img = ImageEnhance.Sharpness(img).enhance(1.3)
    img = ImageEnhance.Color(img).enhance(1.2)
    return np.array(img)

def sharpen_image(img_array):
    #"""Apply a sharpening filter to enhance details"""
    img = Image.fromarray(np.uint8(img_array))
    img = img.filter(ImageFilter.SHARPEN)
    return np.array(img)

def denoise_image(img_array):
    #"""Apply mild denoising to remove artifacts"""
    # Convert to cv2 format
    img = np.array(img_array).astype(np.uint8)
    # Apply mild Gaussian blur to reduce noise
    denoised = GaussianBlur(img, (3, 3), 0.5)
    # Enhance details that might have been lost in blurring
    enhanced = detailEnhance(denoised, sigma_s=10, sigma_r=0.15)
    return enhanced


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