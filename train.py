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

LOAD_MODEL = False

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

        input_captions = np.array([x+[le.transform_word("<NULL>")]*(MAXIMUM_CAPTION_LENGTH-len(x)) for x in input_captions]).astype(np.float32)
        output_captions = one_hot_encode(output_captions, MAXIMUM_CAPTION_LENGTH, MAXIMUM_VOCABULARY_SIZE)
       
        batch_image_features = np.array(batch_image_features, dtype=np.float32)

        x_data = {
            "image_features_input": batch_image_features,
            "captions_input": input_captions
        }

        y_data = {
            "output": output_captions
        }

        yield (x_data, y_data)

def one_hot_encode(data, MAXIMUM_CAPTION_LENGTH, n_classes):
    result = np.zeros((len(data), MAXIMUM_CAPTION_LENGTH, n_classes))
    for i, item in enumerate(data):
        for j, word in enumerate(item):
            result[i, j, word] = 1.0
        for k in range(j+1, MAXIMUM_CAPTION_LENGTH):
            result[i, k, le.transform_word("<NULL>")] = 1.0

    return result

def inference(image_features, plot_attention):
    image_features = np.array([image_features])
    state_h, state_c = initial_state_inference_model.predict(image_features)

    caption = [word_index_map["<START>"]]
    attentions = []

    current_word = None
    for t in range(MAXIMUM_CAPTION_LENGTH):
        caption_array = np.array(caption).reshape(1, -1)
        output, state_h, state_c, attention = inference_model.predict([image_features, caption_array, state_h, state_c])
        attentions.append(attention[0, -1].reshape((14, 14)))

        current_word = np.argmax(output[0, -1])
        caption.append(current_word)

        if current_word == word_index_map["<STOP>"]:
            break
    sentence = [index_word_map[i] for i in caption[1:]]

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

MAXIMUM_VOCABULARY_SIZE = 10000
EMBEDDING_SIZE = 512 # 1024
MAXIMUM_CAPTION_LENGTH = 16

"""
filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
translate_dict = dict((c, " ") for c in filters)
translate_map = str.maketrans(translate_dict)

captions_train_raw, get_image_features_train = load_training_data(MAXIMUM_CAPTION_LENGTH)
captions_val_raw, get_image_features_val = load_validation_data(MAXIMUM_CAPTION_LENGTH)
for i, entry in enumerate(captions_val_raw):
    for j, item in enumerate(entry):
        captions_val_raw[i][j] = item.translate(translate_map).lower()
for i, entry in enumerate(captions_train_raw):
    for j, item in enumerate(entry):
        captions_train_raw[i][j] = item.translate(translate_map).lower()

word_index_map, index_word_map = create_vocabulary(MAXIMUM_VOCABULARY_SIZE, captions_train_raw)
MAXIMUM_VOCABULARY_SIZE = len(word_index_map)

captions_train = encode_annotations(captions_train_raw, word_index_map, MAXIMUM_CAPTION_LENGTH)
captions_val = encode_annotations(captions_val_raw, word_index_map, MAXIMUM_CAPTION_LENGTH)
"""


captions_train_raw, get_image_features_train = load_training_data(MAXIMUM_CAPTION_LENGTH)
captions_val_raw, get_image_features_val = load_validation_data(MAXIMUM_CAPTION_LENGTH)
le = LanguageEncoder(MAXIMUM_VOCABULARY_SIZE)
captions_train = le.fit_transform(captions_train_raw)
captions_val = le.transform(captions_val_raw)
le.save("./models/language.pkl")

def masked_categorical_crossentropy(y_true, y_pred):
    mask_value = le._word_index_map["<NULL>"]
    y_true_id = K.argmax(y_true)
    mask = K.cast(K.equal(y_true_id, mask_value), K.floatx())
    mask = 1.0 - mask
    loss = K.categorical_crossentropy(y_true, y_pred) * mask

    # take average w.r.t. the number of unmasked entries
    return K.sum(loss) / K.sum(mask)

training_model, inference_model, initial_state_inference_model = create_model(le._vocabulary_size, EMBEDDING_SIZE, None, 14*14, 512)
training_model.compile(adam(0.001), loss=masked_categorical_crossentropy, metrics=["accuracy"])

batch_size = 64

def train(epochs=100):
    tbCallback = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False)
    history = training_model.fit_generator(generator(batch_size=batch_size, captions=captions_train, get_image=get_image_features_train), steps_per_epoch=len(captions_train)//batch_size, epochs=epochs, verbose=1, callbacks=[tbCallback])

    training_model.save("./models/sat_train_{0}.h5".format(epochs))
    inference_model.save("./models/sat_inf_{0}.h5".format(epochs))
    initial_state_inference_model.save("./models/sat_inf_init_{0}.h5".format(epochs))

    for key in history.history.keys():
        f = plt.figure()
        data = history.history[key]
        plt.plot(data)
    plt.show()

epochs = input("Number of epochs: ")
epochs = int(epochs)
train(epochs=epochs)
input("done. <read key>")
