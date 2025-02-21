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

from neural_style_transfer import load_preprocess_img,deprocess_img,image_to_bytes,compute_total_loss
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

    model = load_model()
    #content_layer = 'block5_conv2'
    #style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']

    generated_img = tf.Variable(content_img_preprocessed,dtype=tf.float32)
    optimizer = tf.optimizers.Adam(learning_rate=5.0)

    epochs = 50  # Reduce epochs for Streamlit deployment
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(epochs):
      with tf.GradientTape() as tape:
          loss = compute_total_loss(model, content_img_preprocessed, style_img_preprocessed, generated_img)
      grad = tape.gradient(loss, generated_img)
      optimizer.apply_gradients([(grad, generated_img)])

    # Clear memory
      if i % 10 == 0:
        tf.keras.backend.clear_session()
        progress_bar.progress((i + 1)/epochs)
        status_text.text(f"Iteration {i+1}/{epochs}, Loss: {loss.numpy():.2f}")
    
    final_img = deprocess_img(generated_img.numpy())
    st.image(final_img,caption="Generated Image",use_column_width=True)
    st.download_button(label="Download Generated Image",data=image_to_bytes(final_img),file_name="generated_image.jpg",mime="image/jpeg")

if __name__ == "__main__":
    main()

