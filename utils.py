#-*- coding : utf-8 -*-
# coding: utf-8

'''
    Author: Xin Wu
    E-mail: wuxin@icarbonx.com
    June, 2019
'''

import re
import os
import json
import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def python_to_json(python_data, output_path):
    '''Convert python data (tuple, list, dict, etc) into json string'''
    with open(output_path, "w", encoding='utf-8') as write_file:
        # json_str = json.dumps(python_data, indent=4, separators=(',', ':'), cls=NpEncoder)
        # write_file.write(json_str)
        json_str = json.dump(python_data, write_file, ensure_ascii=False, indent=4, cls=NpEncoder)


def read_json_from_file(input_path):
    with open(input_path, "r") as read_file:
        python_data = json.load(read_file)
    return python_data


def Loadlabel(filename):
    f = open(filename, 'r')
    sourceInline = f.readlines()
    f.close()
    labelset = [line.strip() for line in sourceInline]
    return labelset


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"] """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]


def natural_sort(given_list):
    """ Sort the given list in the way that humans expect."""
    given_list.sort(key=alphanum_key)


def get_immediate_subfolder_names(folder_path):
    subfolder_names = [folder_name for folder_name in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, folder_name))]
    natural_sort(subfolder_names)
    return subfolder_names


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def validate_file_format(file_in_path, allowed_format):
    if os.path.isfile(file_in_path) and os.path.splitext(file_in_path)[1][1:] in allowed_format:
        return True
    else:
        return False

def is_image(file_in_path):
    if validate_file_format(file_in_path, ['jpg', 'bmp', 'jpeg', 'JPG', 'JPEG', 'png']):
        return True
    else:
        return False


def cal_aspect_ratio(image_height, image_width):
    if image_height > image_width:
        aspect_ratio = image_width * 1.0 / image_height
    else:
        aspect_ratio = image_height * 1.0 / image_width
    return aspect_ratio


def cal_average_aspect_ratio(bbox_list):
    box_num = len(bbox_list)
    aar = 0.0
    if box_num == 0:
        return aar
    for box in bbox_list:
        box_hight = box[2] - box[0]
        box_weight = box[3] - box[1]
        box_aspect_ratio = cal_aspect_ratio(box_hight, box_weight)
        aar += box_aspect_ratio
    return aar * 1.0 / box_num


def cal_area_ratio(image_height, image_width, box):
    area_image = image_height * image_width
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_ratio = area_box * 1.0 / area_image
    return area_ratio


def cal_lacation_ratio(image_h, image_w, box):

    left, top, right, bottom = box

    center_image_h = image_h / 2
    center_image_w = image_w / 2

    center_box_h = (bottom - top) / 2
    center_box_w = (right - left) / 2

    if (top + center_box_h) <= (image_h - bottom + center_box_h):
        lr_h = (top + center_box_h) * 1.0 / center_image_h
    else:
        lr_h = (image_h - bottom + center_box_h) * 1.0 / center_image_h

    if (left + center_box_w) <= (image_w - right + center_box_w):
        lr_w = (left + center_box_w) * 1.0 / center_image_w
    else:
        lr_w = (image_w - right + center_box_w) * 1.0 / center_image_w

    return lr_h, lr_w


def filter_box(image, box, param_list=[1/8, 1/4, 1/3, 1/3]):

    left, top, right, bottom = box
    image_w, image_h = image.size

    arear = cal_area_ratio(image_h, image_w, box)
    aspectr = cal_aspect_ratio(bottom - top, right - left)
    lr_h, lr_w = cal_lacation_ratio(image_h, image_w, box)

    if ((arear < param_list[0]) and (lr_h < param_list[2] or lr_w < param_list[3])) or \
        (aspectr < param_list[1] and (lr_h < param_list[2] or lr_w < param_list[3])):
        return True
    else:
        return False
