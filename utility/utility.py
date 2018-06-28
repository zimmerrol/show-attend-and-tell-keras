import numpy as np
import utility.coco as coco
import h5py

def load_validation_data(maximum_caption_length):
    coco.set_data_dir("./data/coco")
    coco.maybe_download_and_extract()

    _, _, captions_val_raw = coco.load_records(train=False)

    h5 = h5py.File("image.features.val.VGG19.block5_conv4.h5", "r")
    get_data = lambda i: h5[i]

    return captions_val_raw, get_data


def load_training_data(maximum_caption_length):
    coco.set_data_dir("./data/coco")
    coco.maybe_download_and_extract()

    _, _, captions_train_raw = coco.load_records(train=True)
    
    h5 = h5py.File("image.features.train.VGG19.block5_conv4.h5", "r")
    get_data = lambda i: h5[i]

    return captions_train_raw, get_data


def create_vocabulary(maximum_size, annotations):
    words = dict()
    for annotation_lines in annotations:
        for annotation in annotation_lines:
            for word in annotation.lower().split():
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1
    
    words = [x[0] for x in reversed(sorted(words.items(), key=lambda x: x[1]))]
    words = ["<NULL>", "<START>", "<STOP>"] + words
    words = words[:maximum_size]

    word_index_map = {}
    index_word_map = {}
    for i, word in enumerate(words):
        word_index_map[word] = i
        index_word_map[i] = word

    return word_index_map, index_word_map

def encode_annotations(annotations, word_index_map, maximum_caption_length):
    encoded_annotations = []
    for i, annotation_lines in enumerate(annotations):
        annotations = []
        for j, caption in enumerate(annotation_lines):
            encoded_annotation = []
            for word in caption.split():
                if word.lower() in word_index_map:
                    encoded_annotation.append(word_index_map[word.lower()])
            annotations.append(encoded_annotation)
        encoded_annotations.append(annotations)

    return encoded_annotations