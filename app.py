import os
# Force CPU usage - place this at the very beginning before other imports
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"  # Prevent memory allocation errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings

import streamlit as st
from PIL import Image
# from keras.api.models import load_model
import tensorflow as tf
# Force TensorFlow to use CPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices([], 'GPU')

from model_creation import get_model
@st.cache_resource
def load_model():
   return get_model()

from neural_style_transfer import load_preprocess_img,deprocess_img,image_to_bytes,compute_total_loss,enhance_contrast,match_histograms,np
def main():
  st.title("Neural Style Transfer")
  content_img_file = st.file_uploader("Upload Content Image",type=['jpg','jpeg','png'])
  style_img_file = st.file_uploader("Upload Style Image",type=['jpg','jpeg','png'])

  if content_img_file is not None and style_img_file is not None:
    content_img = Image.open(content_img_file).resize((400,400))
    style_img = Image.open(style_img_file).resize((400,400))

    st.image(content_img,caption="Content Image",use_container_width=True)
    st.image(style_img,caption="Style Image",use_container_width=True)

    rgb_or_rgba = content_img.mode == "RGBA"
    original_alpha = None
    if rgb_or_rgba:
       original_alpha = np.array(content_img.split()[3])

    content_img_preprocessed = load_preprocess_img(content_img)
    style_img_preprocessed = load_preprocess_img(style_img)

    model = load_model()
    #content_layer = 'block5_conv2'
    #style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']

    generated_img = tf.Variable(content_img_preprocessed,dtype=tf.float32)
    optimizer = tf.optimizers.Adam(learning_rate=5.0)

    epochs = 50 # Reduce epochs for Streamlit deployment
    first_phase_epochs = 25
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(first_phase_epochs):
      with tf.GradientTape() as tape:
        loss = compute_total_loss(model, content_img_preprocessed, style_img_preprocessed, generated_img,alpha=0.2, beta=3e3)
      grad = tape.gradient(loss, generated_img)
      optimizer.apply_gradients([(grad, generated_img)])

    # Clear memory
      print(f"Iteration {i}, loss: {loss.numpy()}")
      if i % 10 == 0:
        tf.keras.backend.clear_session()
        progress_bar.progress((i+1)/epochs)
        status_text.text(f"Iteration {i+1}/{epochs}, Loss: {loss.numpy():.2f}")
    
    for i in range(25,50):
      with tf.GradientTape() as tape:
        loss = compute_total_loss(model,content_img_preprocessed,style_img_preprocessed,generated_img,alpha=0.8,beta=1e3)

      grad = tape.gradient(loss,generated_img)
      optimizer.apply_gradients([(grad,generated_img)])
      
      print(f"Refinement phase: {i}, loss: {loss.numpy()}")
      if i%10==0:
         tf.keras.backend.clear_session()
         progress_bar.progress((i+1)/epochs)
         status_text.text(f"Refinement phase: {i+1}/100, Loss: {loss.numpy():.2f}")
    
    final_img = deprocess_img(generated_img.numpy(),rgb_or_rgba,original_alpha)
    #Apply style color matching
    style_array = np.array(style_img.convert('RGB'))

    # Make sure images are converted to numpy arrays before histogram matching
    final_img_arr = np.array(final_img.convert('RGB'))
    
    final_img_arr = match_histograms(final_img_arr,style_array)

    #Enhance contrast
    final_img_arr_enhanced = enhance_contrast(final_img_arr,factor=1.3)
    final_img = Image.fromarray(final_img_arr_enhanced)
    if rgb_or_rgba and original_alpha is not None:
       final_img = final_img.convert("RGBA")
       original_alpha_img = Image.fromarray(original_alpha,"L")
       final_img.putalpha(original_alpha_img)
       
    st.image(final_img,caption="Generated Image",use_container_width=True)
    format_type = "PNG" if rgb_or_rgba else "JPEG"
    st.download_button(
       label="Download Generated Image",
       data=image_to_bytes(final_img,format=format_type),
       file_name=f"generated_image.{format_type.lower()}",
       mime=f"image/{format_type.lower()}")

if __name__ == "__main__":
    main()

