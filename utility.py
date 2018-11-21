"""

Some Tensorflow Utility that interacts with data folders

"""

import tensorflow as tf
import collections
import six


class ImageReader(object):
    
    def __init__(self, image_format="jpeg", channels=3):
        self._decode_data = tf.placeholder(dtype=tf.string)
        self._image_format = image_format
        self._session = tf.Session()
        if self._image_format in ('jpeg', 'jpg'):
            self._decode = tf.image.decode_jpeg(self._decode_data, channels=channels)

        elif self._image_format == 'png':
            self._decode = tf.image.decode_png(self._decode_data, channels=channels)

            
    def _int64_features(self, values):
        """
        Value has to be a list
        """

        if not isinstance(values, collections.Iterable):
            values = [values]

        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


    def _byte64_features(self, value):

        """
        Convert Strings to utf-8 encoding
        
        """

        if isinstance(value, six.string_types):
            value = six.binary_type(value, encoding='utf-8')
        
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def _bytes64_features_google_version(self, values):
        """
        Convert strings and image data to array
        """
        def norm2bytes(value):
            return value.encode() if isinstance(value, str) and six.PY3 else value
        
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))



    def _float64_features(self, value):
        """
        Converting Float values
        Value has to be in a list format
        """

        if not isinstance(value, collections.Iterable):
            value = [value]

        return tf.train.Feature(float_list=tf.train.Feature(value))


    def decode_image(self, image_data):
        image = self._session.run(self._decode, 
                    feed_dict={self._decode_data:image_data})
        return image


    def _read_image_dim(self, image_data):
        image = self.decode_image(image_data)
        return image, image.shape[:2]




    def image_to_tfexample(self, image_data, filename, height, width):

        return tf.train.Example(features=tf.train.Features(
            feature= {
                "image/encoded": self._byte64_features(image_data),
                "image/filename": self._byte64_features(filename),
                "image/height": self._int64_features(height),
                "image/width": self._int64_features(width),
                'image/channels': self._int64_features(3)
            }
        ))

    


    

    



    


