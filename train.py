from model import create_model
from utility.utility import *
import numpy as np
from matplotlib import pyplot as plt
from keras.callbacks import TensorBoard

def generator(batch_size, captions, get_image):
    while True:
        batch_indices = np.random.randint(0, len(captions), size=batch_size, dtype=np.int)
        batch_image_features = np.empty((len(batch_indices), 14*14, 512))
        for i, j in enumerate(batch_indices):
            batch_image_features[i] = get_image(str(j)).value.reshape((14*14, 512))

        batch_captions = [captions[item] for item in batch_indices]

        batch_captions = [x[np.random.randint(0, len(x))][:maximum_caption_length-1] for x in batch_captions]
        input_captions = [[word_index_map["<START>"]] + x for x in batch_captions]
        output_captions = [x + [word_index_map["<STOP>"]] for x in batch_captions]

        # input_captions = one_hot_encode(input_captions, maximum_caption_length, maximum_vocabulary_size)
        output_captions = one_hot_encode(output_captions, maximum_caption_length, maximum_vocabulary_size)

        input_captions = np.array([x+[word_index_map["<NULL>"]]*(maximum_caption_length-len(x)) for x in input_captions]).astype(np.float32)
        # output_captions = np.array([x+[word_index_map["<NULL>"]]*(maximum_caption_length-len(x)) for x in output_captions]).astype(np.float32)
        # input_captions = np.array(input_captions, dtype=np.float32)
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

    return result

def inference(image_features):
    image_features = np.array([image_features])
    state_h, state_c = initial_state_inference_model.predict(image_features)

    caption = [word_index_map["<START>"]]

    current_word = None
    for t in range(maximum_caption_length):
        caption_array = np.array(caption).reshape(1, -1)
        output, state_h, state_c = inference_model.predict([image_features, caption_array, state_h, state_c])
        current_word = np.argmax(output[0, -1])
        caption.append(current_word)

        #if current_word == word_index_map["<STOP>"]:
        #    break
    sentence = [index_word_map[i] for i in caption]

    return " ".join(sentence)

maximum_vocabulary_size = 5000
embedding_size = 1024
maximum_caption_length = 15

filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
translate_dict = dict((c, " ") for c in filters)
translate_map = str.maketrans(translate_dict)

captions_train_raw, get_image_features_train = load_training_data(maximum_caption_length)
captions_val_raw, get_image_features_val = load_validation_data(maximum_caption_length)
for i, entry in enumerate(captions_val_raw):
    for j, item in enumerate(entry):
        captions_val_raw[i][j] = item.translate(translate_map).lower()

word_index_map, index_word_map = create_vocabulary(maximum_vocabulary_size, captions_train_raw)
maximum_vocabulary_size = len(word_index_map)

captions_train = encode_annotations(captions_train_raw, word_index_map, maximum_caption_length)
captions_val = encode_annotations(captions_val_raw, word_index_map, maximum_caption_length)

training_model, inference_model, initial_state_inference_model = create_model(maximum_vocabulary_size, embedding_size, None, 14*14, 512)
training_model.summary()
training_model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])



batch_size = 64

tbCallback = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=16, write_graph=True, write_grads=False, write_images=False)

history = training_model.fit_generator(generator(batch_size=batch_size, captions=captions_train, get_image=get_image_features_train), steps_per_epoch=len(captions_train)//batch_size, epochs=30, verbose=1, callbacks=[tbCallback])

print(history.history.keys())
for key in history.history.keys():
    f = plt.figure()
    data = history.history[key]
    plt.plot(data)
plt.show()

#for _ in range(200):
#    history = training_model.fit_generator(generator(batch_size=batch_size, captions=captions_train, get_image=get_image_features_train), steps_per_epoch=len(captions_train)//batch_size, epochs=1, verbose=1, callbacks=[tbCallback])
#    print("\ninference:")
#    for j in range(10):
#        print("\toutput: {0}".format(inference(get_image_features_val(str(j)).value.reshape(14*14, 512))))
#        print("\ttarget: {0}".format(captions_val_raw[j][0]))
print("saving models...")
training_model.save("sat_train_300.h5")
print("training_model saved.")
inference_model.save("sat_inf_300.h5")
print("inference_model saved.")

input("done. <read key>")
