To Generate tf-record file

python tfrecords.py --image_dir=Images/original --output_dir=Images/tfrecord

This will generate the tfrecord file in tfrecord 

Consuming tf-record file

python read_tf_record.py --tf_record=Images/tfrecord