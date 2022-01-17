import coco_tools
import os
import numpy as np


det_dir = 'detections'
gt_dir = 'groundtruths'
categories = np.array([{'id': 0, 'name': 'garbage'}])


det_files = [os.path.join(det_dir, f) for f in os.listdir(det_dir) if f.endswith('.txt')]
gt_files = [os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.txt')]

image_ids_det = []
image_ids_gt = []
gt_boxes = []
gt_classes = []
det_boxes = []
det_classes = []
det_scores = []

#Read ground truth files to list
for file_name in gt_files:
  file = open(file_name, 'r')
  c = file.read()
  if c == '':
    continue

  content = c.split('\n')
  content.remove('')
  gt_b = []
  gt_c = []
  for line in content:
    split = line.split(' ')
    cl = split[0]
    bbox = np.array([int(x) for x in split[1:]])
    gt_c.append(0) 
    gt_b.append(bbox)
  im_id = file_name.split('/')[-1].replace('.txt', '')
  image_ids_gt.append(im_id) 
  gt_boxes.append(np.array(gt_b))
  gt_classes.append(np.array(gt_c))
  file.close()

#Read detections files to list
for file_name in det_files:
  file = open(file_name, 'r')
  c = file.read()
  if c == '':
    continue

  content = c.split('\n')
  content.remove('')
  det_b = []
  det_c = []
  det_s = []
  for line in content:
    split = line.split(' ')
    cl = split[0]
    score = float(split[1])
    bbox = np.array([int(x) for x in split[2:]])
    det_c.append(0) 
    det_b.append(bbox)
    det_s.append(score)
  im_id = file_name.split('/')[-1].replace('.txt', '')
  image_ids_det.append(im_id) 
  det_boxes.append(np.array(det_b))
  det_scores.append(np.array(det_s))
  det_classes.append(np.array(det_c))
  file.close()

#Convert all lists to numpy arrays
image_ids_gt = np.array(image_ids_gt)
gt_boxes = np.array(gt_boxes)
gt_classes = np.array(gt_classes)
image_ids_det = np.array(image_ids_det)
det_boxes = np.array(det_boxes)
det_classes = np.array(det_classes)

#Convert ground truth list to dict
groundtruth_dict = coco_tools.ExportGroundtruthToCOCO(
      image_ids_gt, gt_boxes, gt_classes,
      categories)

#Convert detections list to dict
detections_list = coco_tools.ExportDetectionsToCOCO(
    image_ids_det, det_boxes, det_scores,
    det_classes, categories)

groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
detections = groundtruth.LoadAnnotations(detections_list)
evaluator = coco_tools.COCOEvalWrapper(groundtruth, detections,
                                       agnostic_mode=False)
metrics, empty = evaluator.ComputeMetrics()
