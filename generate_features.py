import click
import numpy as np
from utility import coco
import h5py
import os
from PIL import Image
from keras.layers import Input
from keras.applications import ResNet50, VGG16, VGG19
from keras.models import Model
from tqdm import tqdm

def setup_model(encoder, layer_name):
    image_input = Input(shape=(224, 224, 3))

    base_model = None
    if encoder == 'vgg16':
        base_model = VGG16(include_top=False, weights='imagenet', input_tensor=image_input, input_shape=(224, 224, 3))
    elif encoder == 'vgg19':
        base_model = VGG19(include_top=False, weights='imagenet', input_tensor=image_input, input_shape=(224, 224, 3))
    else:
        raise ValueError("not implemented encoder type")

    model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)
    return model

def encode_features(model, filenames, image_directory, batch_size=64):
    # calculate encoded features
    generator = data_generator(filenames, image_directory, batch_size=batch_size)
    n_batches = int(np.ceil(len(filenames) / batch_size))

    for i, batch_data in tqdm(enumerate(generator), total=n_batches):
        output = model.predict(batch_data, batch_size=batch_size)
        yield output

def data_generator(filenames, image_directory, batch_size=64):
    n_batches = int(np.ceil(len(filenames) / batch_size))

    for batch_id in range(n_batches):
        batch_image_filenames = filenames[batch_id*batch_size:(batch_id+1)*batch_size]

        batch_images = [None] * len(batch_image_filenames)
        for i, filename in enumerate(batch_image_filenames):
            image = Image.open(os.path.join(image_directory, filename))
            image = image.resize((224, 224)).convert('RGB')
            batch_images[i] = np.asarray(image)
        batch_images = np.array(batch_images)

        x_data = {
            "input_1": batch_images
        }

        yield x_data

@click.command()
@click.option("--data-path", "-d", default="./data/coco/", required=False, type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.option("--encoder", "-e", default="VGG19", required=False, type=click.STRING)
@click.option("--layer-name", "-l", default="block5_conv4", required=False, type=click.STRING)
@click.option("--output-folder", "-o", default=".", required=False, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--batch-size", "-b", default=64, required=False, type=click.INT)
def cmd(data_path, encoder, layer_name, output_folder, batch_size):
    # create data directory if it does not exist
    os.makedirs(data_path, exist_ok=True)
    
    # download the files now if required
    coco.set_data_dir(data_path)
    coco.maybe_download_and_extract()

    # load the data now
    _, filenames_train, captions_train_raw = coco.load_records(train=True)
    _, filenames_val, captions_val_raw = coco.load_records(train=False)

    # encoded the data and save it
    model = setup_model(encoder.strip().lower(), layer_name)

    with h5py.File(os.path.join(output_folder, "image.features.train.{0}.{1}.h5".format(encoder, layer_name)), "w") as h5:
        index = 0
        for batch in encode_features(model, filenames_train, os.path.join(data_path, "train2017"), batch_size=batch_size):
            for item in batch:
                h5.create_dataset(str(index), data=item, compression="lzf")
                index += 1

    with h5py.File(os.path.join(output_folder, "image.features.val.{0}.{1}.h5".format(encoder, layer_name)), "w") as h5:
        index = 0
        for batch in encode_features(model, filenames_val, os.path.join(data_path, "val2017"), batch_size=batch_size):
            for item in batch:
                h5.create_dataset(str(index), data=item, compression="lzf")
                index += 1

# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    cmd()