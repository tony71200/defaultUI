import cv2 as cv
import numpy as np
import time
from PIL import Image
from typing import List
import os

class LoadData:
    def __init__(self, path, img_size=640, slice_width=1024, slice_height=750, overlap_width = 0.0, overlap_height=0.0):
        self._path = path
        self.__dict__.update(locals())
        del self.self  # redundant (and a circular reference)
        self.data = None
        self.cells = []
        self.nf = max(64, len(get_slice_bboxes(3000, 4096, 
                              slice_height=slice_height,
                              slice_width=slice_width,
                              overlap_height_ratio=overlap_height,
                              overlap_width_ratio=overlap_width)))
        # self.loadImage(path)

    def loadImage(self, input_image):
        """
        Require:
        input: array RGB input or path"""
        if isinstance(input_image, str):
            self.data, gray_image = _load_data(input_image, 'pil')
            self.cells = sliceImage(gray_image, self.slice_width, 
                                    self.slice_height, self.overlap_width, self.overlap_height)
            self.nf = len(self.cells)
        elif isinstance(input_image, np.ndarray):
            self.data = input_image
            self.cells = sliceImage(input_image, self.slice_width, 
                                    self.slice_height, self.overlap_width, self.overlap_height)
            self.nf = len(self.cells)
        else:
            raise TypeError('Input type not supported')

    def __len__(self):
        return self.nf

    def __iter__(self):
        self.count = 0
        return self
    
    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        image, pos = self.cells[self.count]
        self.count += 1
        return image, pos
    
    def getImageFull(self):
        return self.data
    
def get_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: int = None,
    slice_width: int = None,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
) -> List[List[int]]:
    """Slices `image_pil` in crops.
    Corner values of each slice will be generated using the `slice_height`,
    `slice_width`, `overlap_height_ratio` and `overlap_width_ratio` arguments.

    Args:
        image_height (int): Height of the original image.
        image_width (int): Width of the original image.
        slice_height (int): Height of each slice. Default 512.
        slice_width (int): Width of each slice. Default 512.
        overlap_height_ratio(float): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio(float): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        auto_slice_resolution (bool): if not set slice parameters such as slice_height and slice_width,
            it enables automatically calculate these params from image resolution and orientation.

    Returns:
        List[List[int]]: List of 4 corner coordinates for each N slices.
            [
                [slice_0_left, slice_0_top, slice_0_right, slice_0_bottom],
                ...
                [slice_N_left, slice_N_top, slice_N_right, slice_N_bottom]
            ]
    """
    slice_bboxes = []
    y_max = y_min = 0

    if slice_height and slice_width:
        y_overlap = int(overlap_height_ratio * slice_height)
        x_overlap = int(overlap_width_ratio * slice_width)
    else:
        raise ValueError("Compute type is not auto and slice width and height are not provided.")

    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes

def sliceImage(image:np.array, slice_width=640, slice_height=640, 
               overlap_width=0.2, overlap_height=0.2, rgb=False):
    if image is None:
        return None
    image_gray = image.copy()
    if len(image.shape) == 3:
        if rgb:
            image_gray = cv.cvtColor(image_gray, cv.COLOR_RGB2GRAY)
        else:
            image_gray = cv.cvtColor(image_gray, cv.COLOR_BGR2GRAY)
    h, w = image_gray.shape[:2]
    mask_range = getMask(image_gray)
    mask_thresh = getThreshold(image_gray, 50)
    concat_image = cv.merge([image_gray, mask_range, mask_thresh])
    # Get the bounding boxes for all the slices in a given image
    h, w = concat_image.shape[:2]
    bboxes = get_slice_bboxes(h, w, 
                              slice_height=slice_height,
                              slice_width=slice_width,
                              overlap_height_ratio=overlap_height,
                              overlap_width_ratio=overlap_width)
    cells = []
    for slice_bbox in bboxes:
        # extract image
        tlx = slice_bbox[0]
        tly = slice_bbox[1]
        brx = slice_bbox[2]
        bry = slice_bbox[3]
        subImage = concat_image[tly:bry, tlx:brx]
        cells.append([subImage, f"{tlx}_{tly}"])
    return cells

def _load_data(path:str, type:["numpy", "pil"]):
    if not os.path.exists(path):
        return None
    image, image_gray = None, None
    try:
        if type == "numpy":
            image = np.asarray(cv.imread(path))
            assert image is not None, f"opencv cannot read image correctly or {path} not exists"
            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        elif type=="pil":
            image = Image.open(path, mode='r')
            image_gray = image.convert("L")
            image = np.asarray(image)
            if (len(image.shape)==3):
                image = image[:,:,::-1]
            image_gray = np.asarray(image_gray)
            assert image is not None, f"Image Not Found {path}, workdir: {os.getcwd()}"
        else:
            raise f"{type} is not supported"
    except:
        assert f"[ERROR]: Failed to load data from {path}"
    return image, image_gray  

def getMask(image:np.array, threshold:tuple = (100, 218)):
    lower = np.array(threshold[0], dtype="uint8")
    upper = np.array(threshold[1], dtype="uint8")
    mask = cv.inRange(image, lower, upper)
    mask = 255 - mask
    return mask

def getThreshold(image:np.array, threshold:int = 50):
    _, mask = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
    return mask