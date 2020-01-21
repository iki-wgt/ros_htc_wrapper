#! /usr/bin/env python
import os
import sys
import mmcv
import rospy
from sensor_msgs.msg import Image
from htc_service_msg.srv import HtcServiceMsg, HtcServiceMsgResponse
from std_srvs.srv import Empty, EmptyResponse
from mmdet.apis import init_detector, inference_detector, show_result, show_result_pyplot
import numpy as np
import pycocotools.mask as maskUtils
import imageio
from rospy.numpy_msg import numpy_msg
from htc_msg.msg import Object
import json

class HtcService:
    def __init__(self):
        self.htc_service_name = 'htc_service'
        self.htc_dir = '/mmdetection'
        self.checkpoint_dir = self.htc_dir
        self.config_dir = os.path.join(self.htc_dir, 'configs', 'htc')
        self.device = 'cuda:0'
        self.htc_detector = None
        self.model_filename = 'htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth'
        self.config_filename = 'htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py'
        self.score_thresh = 0.3
        self.segmentation_publisher = None

    def service_callback(self, request):
        # setup vars for result
        class_name_list = []
        score_list = []
        bbox_list = []
        seg_rle_list = []

        response = HtcServiceMsgResponse()
        print("HTC service has been called")

        # get image out of request
        img = np.frombuffer(request.image.data, dtype=np.uint8).reshape(request.image.height, request.image.width, -1)

        # let detector process image
        result = inference_detector(self.htc_detector, img)

        # extract detection information
        assert isinstance(self.htc_detector.CLASSES, (tuple, list))
        if isinstance(result, tuple):
            bbox_result_full, segm_result_full = result
        else:
            bbox_result_full, segm_result_full = result, None
        bboxes_full = np.vstack(bbox_result_full)

        # collect only detections that have score bigger than score_thresh
        valid_obj_idx_list = np.where(bboxes_full[:, -1] > self.score_thresh)[0]
        bboxes = [bboxes_full[i] for i in valid_obj_idx_list]

        # collect bounding boxes
        for bbox in bboxes:
            bbox_list.append(list(bbox[:4]))

        # collect scores
        for bbox in bboxes:
            score_list.append(bbox[4])

        # collect class labels
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result_full)]
        class_label_id_list_full = np.concatenate(labels)
        class_label_id_list = [class_label_id_list_full[i] for i in valid_obj_idx_list]
        for class_label_id in class_label_id_list:
            class_name_list.append(self.htc_detector.CLASSES[class_label_id])

        # collect segmentation RLEs
        if segm_result_full is not None:
            segms = mmcv.concat_list(segm_result_full)
            for idx in valid_obj_idx_list:
                counts_encoded = segms[idx]['counts']
                counts_decoded = list(counts_encoded)
                segms[idx]['counts'] = counts_decoded
                seg_rle_list.append(json.dumps(segms[idx]))

        # build response message
        for idx in range(len(valid_obj_idx_list)):
            object = Object()
            object.class_name = class_name_list[idx]
            object.score = score_list[idx]
            object.bbox = bbox_list[idx]
            object.rle = seg_rle_list[idx]
            response.objects.append(object)

        return response

    def init_htc_detector(self):
        print("Initialize HTC detector")
        config_file = os.path.join(self.config_dir, self.config_filename)
        checkpoint_file = os.path.join(self.checkpoint_dir, self.model_filename)

        # build the model from a config file and a checkpoint file
        self.htc_detector = init_detector(config_file, checkpoint_file, device=self.device)

    def start_htc_service(self):
        rospy.init_node(self.htc_service_name)
        print('Starting HTC ROS Service')

        # initialize detector
        self.init_htc_detector()

        # start service
        htc_service = rospy.Service(self.htc_service_name, HtcServiceMsg, self.service_callback)

        rospy.spin()

if __name__ == '__main__':
    htc_service = HtcService()
    htc_service.start_htc_service()
