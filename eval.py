from model import create_model
from utility.utility import load_training_data, load_validation_data
from utility.language_encoder import LanguageEncoder
import numpy as np
from matplotlib import pyplot as plt
from keras.callbacks import TensorBoard
import keras.backend as K

from scipy.misc import imresize
import skimage.transform
import matplotlib.pyplot as plt
from keras.models import load_model
from kulc.attention import ExternalAttentionRNNWrapper
import pathlib
from keras.optimizers import adam

MAXIMUM_CAPTION_LENGTH = 16

def generator(batch_size, captions, get_image):
    while True:
        batch_indices = np.random.randint(0, len(captions), size=batch_size, dtype=np.int)
        batch_image_features = np.empty((len(batch_indices), 14*14, 512))
        for i, j in enumerate(batch_indices):
            batch_image_features[i] = get_image(str(j)).value.reshape((14*14, 512))

        batch_captions = [captions[item] for item in batch_indices]

        batch_captions = [x[np.random.randint(0, len(x))][:MAXIMUM_CAPTION_LENGTH-1] for x in batch_captions]
        input_captions = [[le.transform_word("<START>")] + x for x in batch_captions]
        output_captions = [x + [le.transform_word("<STOP>")] for x in batch_captions]

        output_captions = one_hot_encode(output_captions, MAXIMUM_CAPTION_LENGTH, MAXIMUM_CAPTION_LENGTH)
        input_captions = np.array([x+[le.transform_word("<NULL>")]*(MAXIMUM_CAPTION_LENGTH-len(x)) for x in input_captions]).astype(np.float32)

        batch_image_features = np.array(batch_image_features, dtype=np.float32)

        x_data = {
            "image_features_input": batch_image_features,
            "captions_input": input_captions
        }

        y_data = {
            "output": output_captions
        }

        yield (x_data, y_data)

def one_hot_encode(data, maximum_caption_length, n_classes):
    result = np.zeros((len(data), maximum_caption_length, n_classes))
    for i, item in enumerate(data):
        for j, word in enumerate(item):
            result[i, j, word] = 1.0
        for k in range(j+1, maximum_caption_length):
            result[i, k, le.transform_word("<NULL>")] = 1.0

    return result

def inference(image_features, plot_attention):
    image_features = np.array([image_features])
    state_h, state_c = initial_state_inference_model.predict(image_features)

    caption = [le.transform_word("<START>")]
    attentions = []

    current_word = None
    for t in range(MAXIMUM_CAPTION_LENGTH):
        caption_array = np.array(caption).reshape(1, -1)
        output, state_h, state_c, attention = inference_model.predict([image_features, caption_array, state_h, state_c])
        attentions.append(attention[0, -1].reshape((14, 14)))

        current_word = np.argmax(output[0, -1])
        caption.append(current_word)

        if current_word == le.transform_word("<STOP>"):
            break
    sentence = [le._index_word_map[i] for i in caption[1:]]

    if plot_attention:
        print(len(attentions))
        x = int(np.sqrt(len(attentions)))
        y = int(np.ceil(len(attentions) / x))
        _, axes = plt.subplots(y, x, sharex="col", sharey="row")
        axes = axes.flatten()
        for i in range(len(attentions)):
            atn = skimage.transform.pyramid_expand(attentions[i], upscale=16, sigma=20)
            axes[i].set_title(sentence[i])
            axes[i].imshow(atn, cmap="gray")

        plt.show()

    return " ".join(sentence) + " ({0})".format(len(caption)-1)

pathlib.Path('./models').mkdir(exist_ok=True) 

maximum_caption_length = 16

le = LanguageEncoder.load("./models/language.pkl")
captions_val_raw, get_image_features_val = load_validation_data(maximum_caption_length)
captions_val = le.transform(captions_val_raw)

model_id = input("Model ID: ")
model_id = int(model_id)
inference_model = load_model(f"./models/sat_inf_{model_id}.h5", custom_objects={"ExternalAttentionRNNWrapper": ExternalAttentionRNNWrapper})
initial_state_inference_model = load_model(f"./models/sat_inf_init_{model_id}.h5", custom_objects={"ExternalAttentionRNNWrapper": ExternalAttentionRNNWrapper})

while True:
    max_idx = len(captions_val)
    image_idx = input(f"Enter the image index (0-{max_idx}): ")
    image_idx = int(image_idx)
    
    print("output:")
    print("\t {0}".format(inference(get_image_features_val(str(image_idx)).value.reshape(14*14, 512), plot_attention=False)))
    print("target: ")
    for i in range(len(captions_val_raw[image_idx])):
        print("\t{0}".format(captions_val_raw[image_idx][i]))
input("done. <read key>")
