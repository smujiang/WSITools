# from tfrecord.torch.dataset import TFRecordDataset
#
# tfrecord_path = "/tmp/data.tfrecord"
# index_path = None
# description = {"image": "byte", "label": "float"}
# dataset = TFRecordDataset(tfrecord_path, index_path, description)
# loader = torch.utils.data.DataLoader(dataset, batch_size=32)
#
# data = next(iter(loader))
# print(data)


import tensorflow as tf
# Create a dictionary describing the features.
image_feature_description = {
    'loc_x': tf.io.FixedLenFeature([], tf.int64),
    'loc_y': tf.io.FixedLenFeature([], tf.int64),
    'img_w': tf.io.FixedLenFeature([], tf.int64),
    'img_h': tf.io.FixedLenFeature([], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)


tfrecord_fn = "/infodev1/non-phi-data/junjiang/OvaryCancer/Patches_out/1084181_CR02-2502-A2_HE.tfrecord"
template = "../patch_extraction/feature_maps/basic_fm_PL_eval.csv"

raw_dataset = tf.data.TFRecordDataset([tfrecord_fn])

parsed_image_dataset = raw_dataset.map(_parse_image_function)
for image_features in parsed_image_dataset.take(2):
    x = image_features["loc_x"]
    image_raw = image_features['image']



    print(repr(x))
    break


print("DONE")



















