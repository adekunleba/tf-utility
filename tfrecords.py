import tensorflow as tf
import argparse
from utility import ImageReader
import os
import glob
import math
"""
Given an folder of image and segmentation can you  generate the tfRecord


Why Use TFRecords. TFRecords is a binary file format, it's advantage is it takes less space in your memory and allows you to copy
easily and can be more read efficiently from Disk

Integrate easily into different datasets and loading in batch is just easy, also allows to store sequence data or word embeddings

Points:
1. Specify structure of your data before you write to the file just like you do with hdf5
 ---tf.train.Example, tf.train.SequenceExample
2. Serialize the data to disk with TfRecordWriter

Another important class in converting to TFRecord is the tf.train.Features class.
At the end of the day, you generate the following with Features:
byte_list_featares - Wraps a list of datatypes e.g Strings
Int64_list_features - Wraps a list of integer datatypes
Float_list_features - Wraps a list of Float Datatypes

After individual data has been wrapped then the next thing is to also wrap all the data into a conatainer Features, this time a dictionary

"""

NUM_SHARDS = 4


def getImageData(image_file):

    """Decode an image file to image array

    Arguments:
    image_file - png/jpeg file format of image

    return:
    image data (byte format), height, width of data.
    
    """

    # FastGFile retruns a byte format of the data so no need to decode image array here rather
    #decode image array when reading the bytes
    image_data = tf.gfile.FastGFile(image_file, 'rb').read()
    #print(image_data)


    #Build Image Reader for any file format jpeg or png default is jpeg
    reader = ImageReader()

    #Get the Image Array
    image, (height, width) = reader._read_image_dim(image_data)

    #One approach is to convert the image to arrays to bytes,
    image = image.tobytes()

    #Another is to encode images
    return image, height, width



def write_tfRecords(list_of_images):

    """Given list of images write the data to tf_records

    Arguments:
    list_of_images: File list of images

    """

    num_images = len(list_of_images)

    num_per_shard = int(math.ceil(num_images /  float(NUM_SHARDS)))
    reader = ImageReader()


    for share_id in range(NUM_SHARDS):
        OUTPUT_FILENAME = os.path.join(args.output_dir, "{}-of-{}.tfrecord".format(share_id, NUM_SHARDS))
        print("Inserting into shard {}".format(share_id))

        with tf.python_io.TFRecordWriter(OUTPUT_FILENAME) as tfrecord_writer:
            for file in list_of_images:
                image_data, height, width = getImageData(file)
                #print(image_data, file, height, width)

                example = reader.image_to_tfexample(image_data, file, height, width)

                tfrecord_writer.write(example.SerializeToString())



if __name__ == "__main__":
    parsers = argparse.ArgumentParser()
    parsers.add_argument(
        "--image_dir",
        help="Directory for the images"
    )
    parsers.add_argument(
        "--output_dir",
        help="Output directory of tf_record_file"
    )

    args, _ = parsers.parse_known_args()
    print(args.image_dir)

    #list files in image directory
    os.makedirs(args.output_dir, exist_ok=True)
    files = glob.glob(os.path.join(args.image_dir, "*.jpeg"))
    print(len(files))

    write_tfRecords(files)

    