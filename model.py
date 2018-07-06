from keras.layers import Input, Dense, LSTM, TimeDistributed, Embedding, Lambda
from kulc.attention import ExternalAttentionRNNWrapper
from keras.models import Model
import keras.backend as K
import tensorflow as tf

W = 14
H = 14
L = W*H
D = 512

"""
- use features [BS, H, W, D]
- flatten [BS, H*W, D]
- linear transformation [BS, H*W, D]:
    - flatten/reshape -> [BS*H*W, D]
    # Dense(D) -> [BS*H*W, D]
    - reshape -> [BS, H*W, D]

"""
def create_model(vocabulary_size, embedding_size, T, L, D):
    image_features_input = Input(shape=(L, D), name="image_features_input")
    captions_input = Input(shape=(T,), name="captions_input")
    captions = Embedding(vocabulary_size, embedding_size, input_length=T)(captions_input)

    averaged_image_features = Lambda(lambda x: K.mean(x, axis=1))
    averaged_image_features = averaged_image_features(image_features_input)
    initial_state_h = Dense(embedding_size)(averaged_image_features)
    initial_state_c = Dense(embedding_size)(averaged_image_features)

    image_features = TimeDistributed(Dense(D, activation="relu"))(image_features_input)

    encoder = LSTM(embedding_size, return_sequences=True, return_state=True, recurrent_dropout=0.1)
    attented_encoder = ExternalAttentionRNNWrapper(encoder, return_attention=True)

    output = TimeDistributed(Dense(vocabulary_size, activation="softmax"), name="output")

    # for training purpose
    attented_encoder_training_data, _, _ , _= attented_encoder([captions, image_features], initial_state=[initial_state_h, initial_state_c])
    training_output_data = output(attented_encoder_training_data)

    training_model = Model(inputs=[captions_input, image_features_input], outputs=training_output_data)
    
    initial_state_inference_model = Model(inputs=[image_features_input], outputs=[initial_state_h, initial_state_c])
    
    inference_initial_state_h = Input(shape=(embedding_size,))
    inference_initial_state_c = Input(shape=(embedding_size,))
    attented_encoder_inference_data, inference_encoder_state_h, inference_encoder_state_c, inference_attention = attented_encoder(
        [captions, image_features],
        initial_state=[inference_initial_state_h, inference_initial_state_c]
        )
   
    inference_output_data = output(attented_encoder_inference_data)

    inference_model = Model(
        inputs=[image_features_input, captions_input, inference_initial_state_h, inference_initial_state_c],
        outputs=[inference_output_data, inference_encoder_state_h, inference_encoder_state_c, inference_attention]
    )
    
    return training_model, inference_model, initial_state_inference_model