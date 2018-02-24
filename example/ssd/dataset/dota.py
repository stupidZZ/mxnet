from __future__ import print_function
import os
import numpy as np
from imdb import Imdb
import xml.etree.ElementTree as ET
from evaluate.eval_voc import voc_eval
import cv2


class DOTA(Imdb):
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
        super(DOTA, self).__init__('DOTA_' + image_set)
        self.image_set = image_set
        self.data_path = root_path
        self.extension = '.png'
        self.is_train = is_train

        self.classes = ['small-vehicle', 'large-vehicle', 'harbor', 'ship',
                        'ground-track-field', 'soccer-ball-field', 'baseball-diamond',
                        'swimming-pool', 'roundabout', 'tennis-court', 'basketball-court',
                        'plane', 'helicopter', 'bridge', 'storage-tank']


        self.config = {'use_difficult': True,
                       'comp_id': 'comp4',}

        self.num_classes = len(self.classes)
        self.image_set_index = self._load_image_set_index(shuffle)
        self.num_images = len(self.image_set_index)
        if self.is_train:
            self.labels = self._load_image_labels()

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
        """
        find out which indexes correspond to given image set (train or val)

        Parameters:
        ----------
        shuffle : boolean
            whether to shuffle the image list
        Returns:
        ----------
        entire list of images specified in the setting
        """
        image_set_index_folder = os.path.join(self.data_path, 'annotations', self.image_set)
        assert os.path.exists(image_set_index_folder), 'Path does not exist: {}'.format(image_set_index_folder)

        file_list = [f for f in os.listdir(image_set_index_folder) if os.path.splitext(f)[-1] == '.txt']
        #with open(image_set_index_folder) as f:
        #    image_set_index = [x.strip() for x in f.readlines()]
        image_set_index = [os.path.splitext(f)[0] for f in file_list]
        if shuffle:
            np.random.shuffle(image_set_index)
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        full path of this image
        """
        assert self.image_set_index is not None, "Dataset not initialized"
        name = self.image_set_index[index]
        image_file = os.path.join(self.data_path, 'images', name + self.extension)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

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

    def _label_path_from_index(self, index):
        """
        given image index, find out annotation path

        Parameters:
        ----------
        index: int
            index of a specific image

        Returns:
        ----------
        full path of annotation file
        """
        name = self.image_set_index[index]
        label_file = os.path.join(self.data_path, 'annotations', self.image_set, name + '.txt')
        assert os.path.exists(label_file), 'Path does not exist: {}'.format(label_file)
        return label_file

    def _load_image_labels(self):
        """
        preprocess all ground-truths

        Returns:
        ----------
        labels packed in [num_images x max_num_objects x 5] tensor
        """
        temp = []

        # load ground-truth from xml annotations
        for idx, name in enumerate(self.image_set_index):
            label_file = self._label_path_from_index(idx)

            annotation_file = open(label_file, 'r')
            image_source = annotation_file.readline()
            gsd = annotation_file.readline()
            obj_annotations = annotation_file.readlines()
            annotation_file.close()

            image_path = self.image_path_from_index(idx)
            im = cv2.imread(image_path)
            width = im.shape[1]
            height = im.shape[0]
            label = []

            for obj_str in obj_annotations:
                obj_annotation = obj_str.split(' ')
                difficult = int(obj_annotation[-1])
                cls_name = obj_annotation[-2]
                if cls_name not in self.classes:
                    continue
                cls_id = self.classes.index(cls_name)

                xs = [float(x) / width for x in obj_annotation[0:8:2]]
                ys = [float(y) / height for y in obj_annotation[1:8:2]]


                if np.array(xs).max() > 1 or np.array(ys).max() > 1 or np.array(xs).min() < 0 or np.array(ys).min() < 0:
                    pass
                if difficult == 1:
                    pass
                # xml_box = obj.find('bndbox')
                # xmin = float(xml_box.find('xmin').text) / width
                # ymin = float(xml_box.find('ymin').text) / height
                # xmax = float(xml_box.find('xmax').text) / width
                # ymax = float(xml_box.find('ymax').text) / height
                label.append([cls_id, xs[0], ys[0], xs[1], ys[1], xs[2], ys[2], xs[3], ys[3], difficult])
            temp.append(np.array(label))
        return temp

    def evaluate_detections(self, detections):
        """
        top level evaluations
        Parameters:
        ----------
        detections: list
            result list, each entry is a matrix of detections
        Returns:
        ----------
            None
        """
        # make all these folders for results
        result_dir = os.path.join(self.devkit_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        year_folder = os.path.join(self.devkit_path, 'results', 'VOC' + self.year)
        if not os.path.exists(year_folder):
            os.mkdir(year_folder)
        res_file_folder = os.path.join(self.devkit_path, 'results', 'VOC' + self.year, 'Main')
        if not os.path.exists(res_file_folder):
            os.mkdir(res_file_folder)

        self.write_pascal_results(detections)
        self.do_python_eval()

    def get_result_file_template(self):
        """
        this is a template
        VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt

        Returns:
        ----------
            a string template
        """
        res_file_folder = os.path.join(self.devkit_path, 'results', 'VOC' + self.year, 'Main')
        comp_id = self.config['comp_id']
        filename = comp_id + '_det_' + self.image_set + '_{:s}.txt'
        path = os.path.join(res_file_folder, filename)
        return path

    def write_pascal_results(self, all_boxes):
        """
        write results files in pascal devkit path
        Parameters:
        ----------
        all_boxes: list
            boxes to be processed [bbox, confidence]
        Returns:
        ----------
        None
        """
        for cls_ind, cls in enumerate(self.classes):
            print('Writing {} VOC results file'.format(cls))
            filename = self.get_result_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_set_index):
                    dets = all_boxes[im_ind]
                    if dets.shape[0] < 1:
                        continue
                    h, w = self._get_imsize(self.image_path_from_index(im_ind))
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        if (int(dets[k, 0]) == cls_ind):
                            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                    format(index, dets[k, 1],
                                           int(dets[k, 2] * w) + 1, int(dets[k, 3] * h) + 1,
                                           int(dets[k, 4] * w) + 1, int(dets[k, 5] * h) + 1))

    def do_python_eval(self):
        """
        python evaluation wrapper

        Returns:
        ----------
        None
        """
        annopath = os.path.join(self.data_path, 'Annotations', '{:s}.xml')
        imageset_file = os.path.join(self.data_path, 'ImageSets', 'Main', self.image_set + '.txt')
        cache_dir = os.path.join(self.cache_path, self.name)
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self.year) < 2010 else False
        print('VOC07 metric? ' + ('Y' if use_07_metric else 'No'))
        for cls_ind, cls in enumerate(self.classes):
            filename = self.get_result_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imageset_file, cls, cache_dir,
                                     ovthresh=0.5, use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
        print('Mean AP = {:.4f}'.format(np.mean(aps)))

    def _get_imsize(self, im_name):
        """
        get image size info
        Returns:
        ----------
        tuple of (height, width)
        """
        img = cv2.imread(im_name)
        return (img.shape[0], img.shape[1])
