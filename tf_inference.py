import numpy as np
import scipy.misc
import tensorflow as tf
import time

## Settings
FROZEN_FPATH = 'data/frozen.pb' # ADJUST
INPUTE_NODE = 'net/input:0' # ADJUST
OUTPUT_NODE = 'net/fc8/BiasAdd:0' # ADJUST
CLASSES = ['Cat', 'Dog'] # ADJUST
CROP_SIZE = (224, 224) # ADJUST

## Load Frozen graph and create TF Session
graph_def = tf.GraphDef()
with tf.gfile.GFile(FROZEN_FPATH, "rb") as f:    
    graph_def.ParseFromString(f.read())
    
graph = tf.Graph()
with graph.as_default():
    net_inp, net_out = tf.import_graph_def(
        graph_def, return_elements=[INPUTE_NODE, OUTPUT_NODE])
    
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(graph=graph, config=sess_config)


## Calulate Validation Accuracy
import data_provider
image_list, label_list = data_provider.prepare_sample_list(
        '/data/ImageNet/val/','val.txt',classes=[281, 239])

## Warmup
print "Warming up..."
for img_fpath, label in zip(image_list, label_list):
    img = scipy.misc.imread(img_fpath, mode='RGB')
    img = scipy.misc.imresize(img, CROP_SIZE, mode='RGB')
    img = img[None, ...]
    out = sess.run(net_out, feed_dict={net_inp: img})
    
print "Start inferencing..."
image_proc_time = []
correct = 0
for img_fpath, label in zip(image_list, label_list):
    img = scipy.misc.imread(img_fpath, mode='RGB')
    img = scipy.misc.imresize(img, CROP_SIZE, mode='RGB')
    img = img[None, ...]
    st = time.time()
    out = sess.run(net_out, feed_dict={net_inp: img})
    et = time.time() - st
    print str(et * 1000) + " ms"
    image_proc_time.append(et)
    if np.argmax(out) == label:
        correct += 1
        
accuracy = float(correct) / len(image_list)
print('Accuracy: {}'.format(accuracy))
print('There {} pictures inferenced'.format(len(image_proc_time)))
print('Mininum process time per image {} ms'.format(1000. * min(image_proc_time)))
print('Average process time per image {} ms'.format(1000. * sum(image_proc_time)/len(image_proc_time)))
