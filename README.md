# Object Detection Evaluation

Evaluating the results of object detection models in an onerous task, with several metrics such as Mean Average Precision
and Average Recall requiring computation. This repo packages the COCO evaluation metrics by Tensorflow Object Detection API into
a usable Python program.

## Ground truths
The ground truth folder contains text files, one for each image. Each line in a text file is of the format:

`<class> <x1> <y1> <x2> <y2>`

## Detections
The detections folder contains text files as well, one for each image. Each line in a text file is of the format:

`<class> <confidence> <x1> <y1> <x2> <y2>`

## Running the evaluator

1. Navigate to the cocoapi-master/PythonAPI directory.
```
cd cocoapi-master/PythonAPI
```

2. Run the makefile.
```
make
```

3. Copy pycocotools directory back to base directory.
```
mv pycocotools/ ../../
```

4. Open _main.py_ and edit the following variables.

```
det_dir = '/path/to/detections'
gt_dir = '/path/to/groundtruths'
categories = np.array([{'id': 0, 'name': 'className'}, {'id': 1, 'name': 'className'}])
```
5. Run `python3 main.py`

6. Metrics in the following format will be displayed.

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.102
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.333
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.021
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.019
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.105
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.320
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.081
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.141
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.141
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.023
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.151
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.394
 ```
 
 
 ## Credits
 
 Credits to the authors at https://github.com/cocodataset/cocoapi and https://github.com/tensorflow/models/tree/master/research/object_detection. Most of the code is theirs. 
