import time
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
import cv2
try:
    from yolo_onnx.dataset import LoadData
except:
    from dataset import LoadData
# try:
#     from yolo_onnx.utils import nms, draw_detections
# except: 
#     from utils import nms, draw_detections

class OnnxInference:
    def __init__(self,
                 conf_thresh=0.4, 
                 iou_thresh=0.65, 
                 official_nms=False,
                 class_names = ['line', 'scope', 'title', 'error'],
                 colors = [(0, 0, 225), (0, 255, 0), (128, 0, 255), (128, 0, 128)]):
        self.path = None
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.official_nms = official_nms
        self.class_names = class_names
        self.colors = colors
        self.half = False

        self.data = LoadData(None, 640, slice_width=1024, slice_height=750, overlap_width=0.2, overlap_height=0.2)
        self.initialize_model = None

    def initModel(self, model_path, gpu=False):
        print('Loading ONNX Model...')
        providers = [
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ] if gpu else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(
            model_path,
            providers=providers)
        #Get model info
        self.get_input_details()
        self.get_output_details()

        self.has_postprocess = 'score' in self.output_names or self.official_nms
        
    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [
            model_inputs[i].name for i in range(len(model_inputs))
        ]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.input_type = model_inputs[0].type
        if 'float16' in self.input_type:
            self.half = True
        else: self.half = False
        m = self.session.get_modelmeta()
        pass

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def _inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names,
                                   {self.input_names[0]: input_tensor})
        print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs
    
    def prepare_input(self, image, half=False):
        def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
            '''Resize and pad image while meeting stride-multiple constraints.'''
            shape = im.shape[:2]  # current shape [height, width]
            if isinstance(new_shape, int):
                new_shape = (new_shape, new_shape)
            elif isinstance(new_shape, list) and len(new_shape) == 1:
                new_shape = (new_shape[0], new_shape[0])

            # Scale ratio (new / old)
            r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
            if not scaleup:  # only scale down, do not scale up (for better val mAP)
                r = min(r, 1.0)

            # Compute padding
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

            if auto:  # minimum rectangle
                dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

            dw /= 2  # divide padding into 2 sides
            dh /= 2

            if shape[::-1] != new_unpad:  # resize
                im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

            return im, r, (left, top)
        
        image, ratio, dwdh = letterbox(image, new_shape=(self.input_height, self.input_width), auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float16) if half else image.astype(np.float32)
        return im, ratio, dwdh
    
    @staticmethod
    def rescale(ori_shape, boxes:np.ndarray, ratio, dwdh):
        boxes[:, [0,2]] -= dwdh[0]
        boxes[:, [1,3]] -= dwdh[1]
        boxes[:, :4] /= ratio

        # boxes[:, 0].clamp_(0, ori_shape[1])  # x1
        boxes[:, 0].clip(0, ori_shape[1]) # x1
        boxes[:, 1].clip(0, ori_shape[0])  # y1
        boxes[:, 2].clip(0, ori_shape[1])  # x2
        boxes[:, 3].clip(0, ori_shape[0])  # y2

        boxes = boxes.round().astype(np.int32)
        return boxes

    def infer_for_step(self, img_src, pos, 
                        conf_thresh,
                        iou_thresh, 
                        classes,
                        agnostic_nms ,max_det,):
        input_tensor, ratio, dwdh = self.prepare_input(img_src, self.half)

        # Perform inference on the image
        input_tensor = np.ascontiguousarray(input_tensor/255)
        out = self._inference(input_tensor)
        bboxes = []
        for i in range(out[0].shape[0]):
            obj_num = out[0][i]
            boxes = out[1][i]
            scores = out[2][i]
            cls_id = out[3][i]

            boxes = boxes[scores > conf_thresh]
            cls_id = cls_id[scores > conf_thresh]
            scores = scores[scores > conf_thresh]

            img_h, img_w = img_src.shape[:2]
            if len(boxes)> 0 and len(cls_id) > 0 and len(scores) > 0:# NMS
                ori_shape = np.array([(img_h, img_w)] * len(boxes))
                boxes = self.rescale((img_h, img_w), boxes, ratio, dwdh)
                concat = np.concatenate([boxes, np.expand_dims(scores, axis=1), np.expand_dims(cls_id, axis=1), ori_shape], axis=1)
                bboxes.extend(concat)
        return pos, np.array(bboxes)
    
    def detect_single_object(self, image_path:str, max_det = 1000, progress=tqdm):
        self.data.loadImage(image_path)
        results = {}
        for index, (img_src, pos) in enumerate(progress(self.data)):
            pos, result = self.infer_for_step(img_src, pos, 0.3, 0.65, 0,0,0)
            results[pos] = result
        return results
    
    @staticmethod
    def mergeBBoxOffset(pred_results, conf_thresh=0.5, num_class= 4):
        """
        Merge the box following the position of splitted image"""
        merged_boxes = {}
        for pos, value in tqdm(pred_results.items()):
            if value is None or len(value) == 0: continue
            results = value[value[:, 4] > conf_thresh]
            offsetx, offsety = pos.split("_")
            if results.size == 0:
                continue
            offsetx = int(offsetx)
            offsety = int(offsety)
            results[:, [0,2]] += offsetx
            results[:, [1,3]] += offsety
            for cls_id in range(num_class):
                bbox = results[results[:, 5] == cls_id][:, :5]
                if not merged_boxes.__contains__(cls_id):
                    merged_boxes[cls_id]=bbox
                else :
                    merged_boxes[cls_id] = np.concatenate([merged_boxes[cls_id], bbox], axis=0)
        return merged_boxes
    
    @staticmethod
    def joinBB(bb_list:list):
        rects = bb_list.copy()
        # Bool array indicating which initial bounding rect has
        # already been used
        rectsUsed=[False] * len(rects)
        def getXFromRect(item):
            return item[0]
        rects.sort(key=getXFromRect)
        # Array of accepted rects
        acceptedRects = []
        # Merge threshold for x coordinate distance
        xThr = 5

        # Iterate all initial bounding rects
        for supIdx, supVal in enumerate(rects):
            if (rectsUsed[supIdx] == False):

                # Initialize current rect
                currxMin = supVal[0]
                currxMax = supVal[2]
                curryMin = supVal[1]
                curryMax = supVal[3]
                currConf = supVal[4]

                # This bounding rect is used
                rectsUsed[supIdx] = True

                # Iterate all initial bounding rects
                # starting from the next
                for subIdx, subVal in enumerate(rects[(supIdx+1):], start = (supIdx+1)):

                    # Initialize merge candidate
                    candxMin = subVal[0]
                    candxMax = subVal[2]
                    candyMin = subVal[1]
                    candyMax = subVal[3]
                    candConf = subVal[4]

                    # Check if x distance between current rect
                    # and merge candidate is small enough
                    if (candxMin <= currxMax + xThr):

                        # Reset coordinates of current rect
                        # currxMax = candxMax
                        currxMin = min(currxMin, candxMin)
                        currxMax = max(currxMax, candxMax)
                        curryMin = min(curryMin, candyMin)
                        curryMax = max(curryMax, candyMax)
                        currConf = max(currConf, candConf)

                        # Merge candidate (bounding rect) is used
                        rectsUsed[subIdx] = True
                    else:
                        break

                # No more merge candidates possible, accept current rect with format (x1, y1, x2, y2, conf)
                acceptedRects.append([currxMin, curryMin, currxMax, curryMax, currConf])
        return acceptedRects 

    def postProcessing(self, bbox_infor:dict):
        """Post processing function to merge the close bboxes in 16 splited image"""
        mergedBboxInfor = bbox_infor.copy()
        for key, value in bbox_infor.items():
            bbox = self.joinBB(value.tolist())
            mergedBboxInfor[key]=bbox
        return mergedBboxInfor
    
    def data_lengh(self):
        '''return length of dataset'''
        return len(self.data)
        
    # def detect_batch_object(self, image_path, batch = 16, progress=tqdm):
    #     self.data.loadImage(image_path)
    #     resize_data = []
    #     infer_data = []
    #     resize_datas = []
    #     infer_datas = []
    #     for index, (img_src, pos) in enumerate(progress(self.data)):
    #         input_tensor, ratio, dwdh = self.prepare_input(img_src)
    #         if index // batch in range(batch): 
    #             resize_data.append((ratio, dwdh, pos))
    #             infer_data.append(input_tensor)
    #     pass

if __name__ == '__main__':
    model_path = r"weights\scope6m_f16.onnx"
    # image_path = r"D:\001_dataset\train\20230516-1696-ALL-2_F-01-04_0-0.png"
    image_path = r"D:\000_dataset\000_NewDataset\20230516-1696-ALL-2\F-04-04.bmp"
    # image_path = r"D:\000_dataset\SubImg\20230516-1696-ALL-2\F-04-04_1-2.png"
    detector = OnnxInference()
    detector.initModel(model_path, gpu=False)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # b = detector.infer_for_step(image, "0_0", 0.35, 0.65, classes=['line'], agnostic_nms=True, max_det=1000)
    # print(b)
    detector.detect_single_object(image_path)
    # detector.detect_batch_object(image_path)

    
#     # # i = draw_detections(image, o[1], o[2], o[3])
#     # cv2.namedWindow("i", cv2.WINDOW_GUI_EXPANDED)
#     # cv2.imshow("i", i)
#     # cv2.waitKey()
#     # detector.detect_single_object(image_path)
    

    