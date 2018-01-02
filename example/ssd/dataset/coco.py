from __future__ import print_function
import os
import numpy as np
from imdb import Imdb
import xml.etree.ElementTree as ET
from evaluate.eval_voc import voc_eval
import cv2
from .pycocotools.coco import COCO
from .pycocotools.cocoeval import COCOeval


class coco(Imdb):
    """
    Implementation of Imdb for Pascal VOC datasets

    Parameters:
    ----------
    image_set : str
        set to be used, can be train, val, trainval, test
    year : str
        year of dataset, can be 2007, 2010, 2012...
    devkit_path : str
        devkit path of VOC dataset
    shuffle : boolean
        whether to initial shuffle the image list
    is_train : boolean
        if true, will load annotations
    """
    def __init__(self, image_set, root_path, shuffle=False, is_train=False):
        super(coco, self).__init__('COCO' + '_' + image_set)
        self.image_set = image_set
        self.data_path = root_path
        self.extension = '.jpg'
        self.is_train = is_train
        self.coco = COCO(self._get_ann_file())


        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats

        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict([(self._class_to_coco_ind[cls], self._class_to_ind[cls])
                                            for cls in self.classes[1:]])



        view_map = {'minival2014': 'val2014',
                    'minitrain2017': 'train2017',
                    'valminusminival2014': 'val2014',
                    'test-dev2015': 'test2015'}
        self.data_name = view_map[image_set] if image_set in view_map else image_set

        self.config = {'use_difficult': True,
                       'comp_id': 'comp4',}

        self.num_classes = len(self.classes)
        self.image_set_index = self._load_image_set_index(shuffle)
        self.num_images = len(self.image_set_index)
        if self.is_train:
            self.labels = self._load_image_labels()


    def _get_ann_file(self):
        """ self.data_path / annotations / instances_train2014.json """
        prefix = 'instances' if 'test' not in self.image_set else 'image_info'
        return os.path.join(self.data_path, 'annotations',
                            prefix + '_' + self.image_set + '.json')


    @property
    def cache_path(self):
        """
        make a directory to store all caches

        Returns:
        ---------
            cache path
        """
        cache_path = os.path.join(os.path.dirname(__file__), '..', 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path

    def _load_image_set_index(self, shuffle):
        """ image id: int """
        image_ids = self.coco.getImgIds()
        return image_ids


    def image_path_from_index(self, index):
        index = self.image_set_index[index]
        """ example: images / train2014 / COCO_train2014_000000119993.jpg """
        if '2017' in self.data_name:
            filename = '%012d.jpg' % index
        else:
            filename = 'COCO_%s_%012d.jpg' % (self.data_name, index)
        image_path = os.path.join(self.data_path, 'images', self.data_name, filename)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path


    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        ground-truths of this image
        """
        assert self.labels is not None, "Labels not processed"
        return self.labels[index]

    def _load_image_labels(self):
        """
        preprocess all ground-truths

        Returns:
        ----------
        labels packed in [num_images x max_num_objects x 5] tensor
        """
        temp = []

        # load ground-truth from xml annotations
        for i, idx in enumerate(self.image_set_index):
            im_ann = self.coco.loadImgs(idx)[0]
            width = im_ann['width']
            height = im_ann['height']

            annIds = self.coco.getAnnIds(imgIds=idx, iscrowd=False)
            objs = self.coco.loadAnns(annIds)

            if i == 896:
                pass


            # sanitize bboxes
            valid_objs = []
            for obj in objs:
                x, y, w, h = obj['bbox']
                x1 = np.max((0, x))
                y1 = np.max((0, y))
                x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
                if obj['area'] > 0 and x2 > x1 and y2 > y1:
                    obj['clean_bbox'] = [x1, y1, x2, y2]
                    valid_objs.append(obj)
            objs = valid_objs

            label = []
            for obj in objs:
                iscrowd = obj['iscrowd']
                cls_id = self._coco_ind_to_class_ind[obj['category_id']] - 1

                xmin = float(obj['clean_bbox'][0]) / width
                ymin = float(obj['clean_bbox'][1]) / height
                xmax = float(obj['clean_bbox'][2]) / width
                ymax = float(obj['clean_bbox'][3]) / height

                label.append([cls_id, xmin, ymin, xmax, ymax, iscrowd])

            temp.append(np.array(label))

        return temp

