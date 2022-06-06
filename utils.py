# Utility functions

import cv2 as cv
import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import operator
import itertools
from tqdm import tqdm  


def image_text_crop(images, filenames, ground_truth, one_file=True, result_folder='./results'):
    """
    Crops and saves images based on bounding box ground truth for each text region.
    Creates text file with corresponding annotation.

    Parameter:
    - images: loaded images
    - filenames: list of image filenames with extension
    - groun_truth: list of gt tuples first text annotation, second np.array of 
    left top and bottom right coodinates, format: ('text', [[tl,tl],[br,br]])
    """

    # test if there are not more gts than images
    # else the for loop will never get to those exceeding image count
    gt_length = len(ground_truth)
    if len(images) > gt_length:
        images = images[:gt_length]

    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    
    all_texts = []
    for i, img in tqdm(enumerate(images)):
        name, ext = os.path.splitext(filenames[i])

        # count regions in one image - used for file naming purposes
        region = 1
        
        for text, bbox in ground_truth[i]:
            # select image within coordinates (bbox)
            cropped = img[bbox[0,1]:bbox[1,1], bbox[0,0]:bbox[1,0]]

            # create image file:
            # name in format "original-00region.ext"
            new_name = name + '-' + str(region).zfill(3)
            ext_tif = '.tif'

            cv.imwrite(os.path.join(result_folder, new_name + ext_tif), cropped)
            # create  text annotation file(s)
            if one_file:
                all_texts.append(new_name + ext_tif + '\t' + text)
            else:
                # one file for each image with word
                with open(os.path.join(result_folder, new_name + '.gt.txt'), 'w') as f:
                    f.write(text)
            region += 1
    
    if one_file:
        with open(os.path.join(result_folder, 'gt.txt'), 'w') as f:
            for line in all_texts:
                f.writelines(line+'\n')
            

def shrink_all(images, width):
    """
    Returns list of resized images to given width or leaves the ones already smaller than width. Aspects ratio.
    """
    scaled = []
    
    for image in images:
        if image.shape[1] > width:
            ratio = width / image.shape[1]
            height = int(image.shape[0] * ratio)
            new_size = (width, height)  
            scaled.append(cv.resize(image, new_size, interpolation=cv.INTER_AREA))
        else:
            scaled.append(image)
    return scaled

def bounding_rectangle(coordinates):
    """
    Returns top left and bottom right coordinates of a rectangle,
    counted from a polygon.
    """
    x, y = zip(*coordinates)

    return np.array([[int(min(x)), int(min(y))], [int(max(x)), int(max(y))]])

def group_text(lst):
    """
    Matches corresponing text words.
    Returns list of tuples, where tuple contains sum of first element of list 
    and tuple of indeces that repeated for the key (key is the last third 
    element of given list). 
    """
    grouped = []
    key = lambda x: x[2]

    for k, g in itertools.groupby(sorted(lst, key=key), key):
        list_data = list(zip(*g))
        grouped.append((sum(list_data[0]), list_data[1]))

    return grouped
    
    
def average(tupl, length=None):
    if length is None:
        return sum(tupl) / len(tupl)
    else:
        return sum(tupl) / length

# -------------------------- IOU -------------------------- 
def iou(pred_box, gt_box):
    """
    Computes iou (intersection over union) of two bounding boxes.

    Parameters:
    - pred_box: predicted coordinates (top left, bottom right) of a bounding box rectangle
    - gt_box: ground thruth coordinates (top left, bottom right) of a bounding box rectangle
    """

    # find intersection rectangle coordinates
    x_left = max(pred_box[0][0], gt_box[0][0])
    x_right = min(pred_box[1][0], gt_box[1][0])
    y_top = max(pred_box[0][1], gt_box[0][1])
    y_bottom = min(pred_box[1][1], gt_box[1][1])

    if x_right < x_left or y_bottom < y_top:
        return 0
    
    # compute intersection area
    intersection = (x_right - x_left) * (y_bottom - y_top)

    # compute union area
    pred_area = (pred_box[1][0] - pred_box[0][0]) * (pred_box[1][1] - pred_box[0][1]) 
    gt_area = (gt_box[1][0] - gt_box[0][0]) * (gt_box[1][1] - gt_box[0][1]) 
    union  = pred_area + gt_area - intersection

    # return iou
    return  intersection/union


def iou_image(pred_boxes, gt_boxes):
    """
    Computes iou (intersection over union) for all text regions in one image.
    Each parameter shall contain a list of two coordinates - (top left, bottom 
    right) of a bounding box rectangle.
    Returns list of tuples - tuple of best iou value, index of predicted 
    bounding box and index of ground truth bounding box.
    """
    ious = []

    # have to determine which prediction bounding box contains same (similar) 
    #  text region as ground truth bounding box
    # find and save the best iou for a prediction box and gt box
    for pred_ind, pred in enumerate(pred_boxes):
        max_iou = 0
        max_ind = 0
        for gt_ind, gt in enumerate(gt_boxes):
            iou_value = iou(pred, gt)
            if (iou_value > max_iou):
                max_iou = iou_value
                max_ind = gt_ind

        # match words from prediction and ground thruth (indeces)     
        ious.append((max_iou, pred_ind, max_ind))

    return ious

# -------------------------- XML parsing -------------------------- 
def read_gt_ctw_test(data, scaling_ratio=1):
    """
    SCUT-CTW1500 dataset (test labels parser)
    """
    # one line = one bounding polygon : list of coordinates, each separated by commas, last is the text inside 
    # there are #### before each text, two additional ## no text recognized


    annotations = []
    with open(data, "r") as file:
        for line in file:
            line = line.rstrip('\n')
            text = line.split("####")
            label = text[-1]
            coordinates = text[0].split(",")[:-1]
            c = [int(i) for i in coordinates]
            minX = min(c[::2])*scaling_ratio
            maxX = max(c[::2])*scaling_ratio
            minY = min(c[1::2])*scaling_ratio
            maxY = max(c[1::2])*scaling_ratio

            bbox_coords = np.array( [[minX, minY], [maxX, maxY]] )
            annotations.append((label, bbox_coords))

    return annotations


def read_gt_ctw_train(xml_file, scaling_ratio=1):
    """
    SCUT-CTW1500 dataset (XML - train labels parser)
    Returns ground truth in a tuple - first contains coordinates (8 numbers), second word (string).
    If image was previously scaled, one might need to scale also gt coordinates by given ratio.
    """
    gt = []

    tree = ET.parse(xml_file)
    root = tree.getroot()

    # get values in this order: height, left coordinate, top coordinate, width
    for i, bbox in enumerate(root[0].findall('box')):
        # create list of integers with bounding box values, sort by attribute name
        # in case in different document there is a different order of attributes
        bbox_integer = [int(val) for key, val in sorted(bbox.attrib.items(), key = lambda el: el[0])]
        
        # calculate bottom coordinate of bounding rectangle x+width, y+height
        x_right= int((bbox_integer[1] + bbox_integer[3]) * scaling_ratio)
        y_bottom = int((bbox_integer[2] + bbox_integer[0]) * scaling_ratio)
        x_left = int(bbox_integer[1] * scaling_ratio)
        y_top = int(bbox_integer[2] * scaling_ratio)

        bbox_coords = np.array([[x_left, y_top], [x_right, y_bottom]])

        # get label
        label = root[0][i].find('label').text

        # create list of labels and corresponding boundin boxes
        gt.append((label, bbox_coords))

    return gt
    
    
def read_gt_kaist(xml_file, scaling_ratio=1):
    """
    KAISTdetectiondataset dataset (XML - labels parser)
    Returns ground truth in a tuple - first contains coordinates (8 numbers), second word (string).
    If image was previously scaled, one might need to scale also gt coordinates by given ratio.
    """
    gt = []

    tree = ET.parse(xml_file)
    root = tree.getroot()
    print(root[0][2])
    # get values in this order: height, width, x (left) coordinate, y (top) coordinate
    for i, bbox in enumerate(root[0][2].findall('word')):
        # create list of integers with bounding box values, sort by attribute name
        # in case in different document there is a different order of attributes
        bbox_integer = [int(val) for key, val in sorted(bbox.attrib.items(), key = lambda el: el[0])]
        
        # calculate bottom coordinate of bounding rectangle x+width, y+height
        x_right= int((bbox_integer[2] + bbox_integer[1]) * scaling_ratio)
        y_bottom = int((bbox_integer[3] + bbox_integer[0]) * scaling_ratio)
        x_left = int(bbox_integer[2] * scaling_ratio)
        y_top = int(bbox_integer[3] * scaling_ratio)

        bbox_coords = np.array([[x_left, y_top], [x_right, y_bottom]])

        # get label
        label = ""
        for char in root[0][2][i].findall('character'):
            ch = char.get('char')
            print(char)
            label += ch
        # create list of labels and corresponding boundin boxes
        gt.append((label, bbox_coords))

    return gt
    
    
# -------------------------- Text Comparision -------------------------- 
def compare_text_cer(text, special_characters=False, case_sensitive=False):
    """
    Parameters:
    - text: tuple of (ground_truth, predicted)
    - special_characters=False: if False then ignores all characters except alphanumeric
    - case_sensitive=False: if False then interprets text as lowercase 
    """
    gt, pred = text
    # remove special characters and case sensitivity if necessary
    if ~special_characters:
        text_gt = "".join(char for char in gt if (char.isalnum() or char.isspace()))
        text_pred = "".join(char for char in pred if (char.isalnum() or char.isspace()))
    if ~case_sensitive:
        text_gt = text_gt.lower()
        text_pred = text_pred.lower()
    
    # devide text region on single words
    # because one text_... can contain more than one word
    # and there is a possibility that these words are not in correct order
    # space ic the separator
    words_gt = text_gt.split(" ")
    words_pred = text_pred.split(" ")

    # list of words that are corresponding (based on levenshtein distance)
    # and cer value. (=tuple of three elements)
    # for every predicted word find its corresponding gt wordle
    corresponding_words = []
    for word_pred in words_pred:    
        min_dist = (1000, (0, 0))
        min_gt_word = ""                  
        for word_gt in words_gt: 
            l_dist = levenshtein_distance(word_gt, word_pred)
            if l_dist[0] < min_dist[0]:
                min_dist = l_dist
                min_gt_word = word_gt
        # count normalized cer (the result will be from 0 to 1), 1 is the worst
        # for computation we devide Levenshtein dist. by sum 
        # of the length of the word and count of insertions performed
        if len(min_gt_word) > 0 and len(word_pred) > 0:
            cer = min_dist[0] / (len(min_gt_word) + min_dist[1][2])
        else:
            cer = 1
        corresponding_words.append((min_gt_word, word_pred, cer))

    return sorted(corresponding_words)
    
    
def levenshtein_distance(u, v):
    # This function comes from console application called xer.
    # Its code is available from https://github.com/jpuigcerver/xer/blob/master/xer.
    # Returns Levenshtein distance and tuple of performed operations (substitution, deletion, insertion)
    prev = None
    curr = [0] + list(range(1, len(v) + 1))
    # Operations: (SUB, DEL, INS)
    prev_ops = None
    curr_ops = [(0, 0, i) for i in range(len(v) + 1)]
    for x in range(1, len(u) + 1):
        prev, curr = curr, [x] + ([None] * len(v))
        prev_ops, curr_ops = curr_ops, [(0, x, 0)] + ([None] * len(v))
        for y in range(1, len(v) + 1):
            delcost = prev[y] + 1
            addcost = curr[y - 1] + 1
            subcost = prev[y - 1] + int(u[x - 1] != v[y - 1])
            curr[y] = min(subcost, delcost, addcost)
            if curr[y] == subcost:
                (n_s, n_d, n_i) = prev_ops[y - 1]
                curr_ops[y] = (n_s + int(u[x - 1] != v[y - 1]), n_d, n_i)
            elif curr[y] == delcost:
                (n_s, n_d, n_i) = prev_ops[y]
                curr_ops[y] = (n_s, n_d + 1, n_i)
            else:
                (n_s, n_d, n_i) = curr_ops[y - 1]
                curr_ops[y] = (n_s, n_d, n_i + 1)
    return curr[len(v)], curr_ops[len(v)]
    
    # # count correct characters in word
    # correct = 0
    # cols = len(word1)+1

    # # distance matrix
    # rows = len(word2)+1
    # d = np.zeros((rows, cols), dtype=np.int32)
    # d[0,:] = np.arange(cols)
    # d[:,0] = np.arange(rows)

    # for i in range(1, rows):    
    #     for j in range(1, cols):
    #         if word1[j-1] != word2[i-1]:
    #             cost = 1
    #         else:
    #             cost = 0    
    #         values = (d[i-1,j] + 1, d[i,j-1] + 1, d[i-1,j-1] + cost)
    #         d[i, j] = min(values)
    #         correct +=1 if values.index(d[i, j]) else correct

    # return d.flat[-1], correct


# -------------------------- Final Visualisation -------------------------- 
def plot_results(image, ground_truth, predicted, size=15):
    """
    Return plot with image and both predicted and ground truth bounding boxes 
    and corresponding labels.

    Parameters:
    - image: loaded image
    - ground_truth: tuple of label and top left and bottom right coordinates
      (array of two arrays each of two ints)
    - predicted: tuple of label and top left and bottom right coordinates
      (array of two arrays each of two ints)
    - size=15: size of plot
    """
    # Create figure and axes
    figure, ax = plt.subplots(figsize=(size, size))

    # Display the image
    ax.imshow(image, cmap=plt.get_cmap('gray'))
    ax.axis('off')

    for label, bbox  in ground_truth:
        topleft = bbox[0]
        height = bbox[1,1] - bbox[0,1]
        width = bbox[1,0] - bbox[0,0]

        # create and add rectangle
        rect = patches.Rectangle((topleft), width, height, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

        # add labels
        ax.text(topleft[0]+width, topleft[1],label,verticalalignment='top', color='g',fontsize=13, bbox=dict(facecolor='g', alpha=0.2, edgecolor='g'))

    for label, bbox  in predicted:
        topleft = bbox[0]
        height = bbox[1,1] - bbox[0,1]
        width = bbox[1,0] - bbox[0,0]

        # create and add rectangle
        rect = patches.Rectangle((topleft), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # add labels
        ax.text(topleft[0]-2, topleft[1]-5,label,verticalalignment='top', color='r',fontsize=13, bbox=dict(facecolor='r', alpha=0.2, edgecolor='r'))
    
    # smaller white borders
    plt.subplots_adjust(left=0, bottom=0.1, right=1, top=0.9, wspace=0, hspace=0)

    return plt   