import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np

# Define the custom layers
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.positional_encoding = self.get_positional_encoding(max_len, embed_dim)

    def get_positional_encoding(self, max_len, embed_dim):
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(embed_dim)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # Apply sin to even indices
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # Apply cos to odd indices
        return tf.constant(angle_rads, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.positional_encoding[: tf.shape(inputs)[1], :]

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    with tf.keras.utils.custom_object_scope({
        "PositionalEncoding": PositionalEncoding,
        "TransformerBlock": TransformerBlock
    }):
        model = load_model("transformer_model_final.h5")
    with open("tokenizerfinal.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

def predict_response(input_text, model, tokenizer, num_predictions=20):
    # Dynamically determine max_length from model
    max_length = model.input_shape[1]
    current_text = input_text

    for _ in range(num_predictions):
        # Tokenize and pad the input sequence
        input_sequence = tokenizer.texts_to_sequences([current_text])
        input_sequence = tf.keras.preprocessing.sequence.pad_sequences(input_sequence, maxlen=max_length, padding="post")

        # Predict the next word
        predicted_probs = model.predict(input_sequence, verbose=0)
        predicted_index = tf.argmax(predicted_probs, axis=-1).numpy()[0]
        predicted_word = tokenizer.index_word.get(predicted_index, "")
        if not predicted_word:  # Stop if no valid prediction
            break

        # Update the input text with the predicted word
        current_text += f" {predicted_word}"
        current_text = " ".join(current_text.split()[-max_length:])
    return current_text

# Streamlit app UI
st.title("Chatbot with Transformer Model")
st.markdown("Type a message below to chat with the bot.")

user_input = st.text_input("Your Message:", "")

if user_input:
    response = predict_response(user_input, model, tokenizer)
    st.text_area("Chatbot Response:", response, height=200)
