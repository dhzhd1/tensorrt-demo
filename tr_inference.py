import numpy as np
import scipy.misc
import tensorrt as trt
import time

## Settings
ENGINE_FPATH = 'data/engine.plan' # ADJUST
CLASSES = ['Cat', 'Dog'] # ADJUST
CROP_SIZE = (224, 224) # ADJUST

engine = trt.lite.Engine(PLAN=ENGINE_FPATH)

## Calculate validation accuracy
import data_provider
image_list, label_list = data_provider.prepare_sample_list(
        '/data/ImageNet/val/','val.txt', classes=[281, 239])

##Warm up
#print "Warming up..."
#for img_fpath, label in zip(image_list, label_list):
#    img = scipy.misc.imread(img_fpath, mode='RGB')
#    img = scipy.misc.imresize(img, CROP_SIZE, mode='RGB')
#    img = img[None, ...]
#    out = engine.infer(img)



## Start Inferencing
print "Starting inferencing...."


process_time = []
correct = 0
for img_fpath, label in zip(image_list, label_list):
    img = scipy.misc.imread(img_fpath, mode='RGB')
    img = scipy.misc.imresize(img, CROP_SIZE, mode='RGB')
    img = img[None, ...]
    st = time.time()
    out = engine.infer(img)
    process_time.append(time.time() - st)
    print str(1000 * process_time[-1]) + ' ms' 
    if np.argmax(out[0]) == label:
        correct += 1

# for inf_time in process_time:
#    print str(1000 * inf_time) + ' ms'
        
accuracy = float(correct) / len(image_list)
print('Accuracy: {}'.format(accuracy))
print('Image processed {}'.format(len(process_time)))
print('Minimun process time {} ms'.format(1000. * min(process_time)))
print('Average process time {} ms'.format(1000. * sum(process_time)/len(process_time)))
