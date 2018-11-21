import tensorflow as tf
import argparse
import glob
import os

#Reading TfRecord files i s quite simple



def makeImage(data_record):
    #Understand tf.FixedLenFeatures and tf.VarLenFeature
    feature_set = {
         "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/filename":tf.FixedLenFeature([], tf.string),
        "image/height" : tf.FixedLenFeature([], tf.int64),
        "image/width" : tf.FixedLenFeature([], tf.int64),
        "image/channels": tf.FixedLenFeature([], tf.int64)
    }
    features = tf.parse_single_example(data_record, features=feature_set)

    #TODO: To convert converting the decoded images to 3 DImensional

    image = tf.decode_raw(features["image/encoded"], tf.uint8)
    
    height = tf.cast(features["image/height"], tf.int32)
    width =  tf.cast(features["image/width"], tf.int32)
    depth = tf.cast(features["image/channels"], tf.int32)
    # #image.set_shape([width * height * depth])
    image = tf.cast(
        tf.transpose(tf.reshape(image, [height, width, depth]), [1, 2, 0]), tf.float32
    )
    filename = features["image/filename"]
    #This is the logic of whatever you want
    return image, filename, height, width, depth



def generate(list_data):
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(list_data)#repeat()- use repeat here
    dataset = dataset.map(makeImage)
    #dataset = dataset.shuffle(buffer_size=100000)
    #you put if you want to generate batches
    #dataset = dataset.batch(batchsize=32)

    #Or make repeat here 
    #dataset = dataset.reapeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    #This is to test the iteration
    session = tf.Session()
    #print(sum(1 for _ in tf.python_io.tf_record_iterator(args.tf_records)))
    for n in range(10):
        value = session.run(next_element)
        print(value)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tf_record",
        help="TF Record Directory"
    )
    args, _ = parser.parse_known_args()

    files = glob.glob(os.path.join(args.tf_record, "*.tfrecord"))
    print("TF Records has {}".format(len(files)))
    generate(files)