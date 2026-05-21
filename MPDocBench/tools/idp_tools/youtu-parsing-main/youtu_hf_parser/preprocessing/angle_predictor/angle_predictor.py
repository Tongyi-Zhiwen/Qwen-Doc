import os
import sys
import time
import math
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from queue import Queue
from collections import deque
from numba import jit

# Add parent directory to sys.path for local imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from .model_structure.model import PTD


@jit(nopython=True)
def pse_bfs_numba(pred, kernals, queue_arr, queue_head, queue_tail):
    """
    Numba-accelerated BFS core logic for Progressive Scale Expansion (PSE).
    
    Args:
        pred: Current prediction result (int32 2D array)
        kernals: 3D array (num_kernels, h, w) storing all kernels
        queue_arr: Pre-allocated 1D array simulating queue, format [x, y, label_id]
        queue_head, queue_tail: Queue head and tail pointers
    
    Returns:
        Updated prediction array with expanded regions
    """
    h, w = pred.shape
    kernal_num = kernals.shape[0]
    
    # 4-neighborhood offsets
    dx = np.array([-1, 1, 0, 0])
    dy = np.array([0, 0, -1, 1])

    # Expand from second-to-last kernel outward
    for k_idx in range(kernal_num - 2, -1, -1):
        curr_kernel = kernals[k_idx]
        
        # Current layer queue processing
        # Note: We use a single large array to simulate queue with pointer movement
        # New points are appended to tail, current layer continues processing until no expansion possible
        
        # PSE logic: Complete BFS for current kernel layer before moving to next kernel layer
        # We can reuse the same queue since BFS guarantees layer ordering
        
        # To strictly follow PSE layer-by-layer expansion (complete kernel_i before kernel_i-1),
        # we need to know where current layer boundary is
        # Simple approach: process while queue not empty, but limit validity check to curr_kernel
        # Since we go from small to large kernels, larger curr_kernel contains smaller ones
        
        # Key optimization: To avoid infinite loops or logic errors, Numba version typically
        # processes all possible expansions for current layer at once
        # Logic:
        # 1. Queue contains boundary points from previous layer
        # 2. Try to expand in four directions
        # 3. If expansion point is within curr_kernel and unmarked, mark it and add to queue
        
        # To distinguish layers, record current layer tail position as end point
        current_layer_end = queue_tail
        
        while queue_head < current_layer_end:
            # Pop from queue
            x = queue_arr[queue_head, 0]
            y = queue_arr[queue_head, 1]
            l = queue_arr[queue_head, 2]
            queue_head += 1
            
            # Check 4 directions
            for j in range(4):
                tmpx = x + dx[j]
                tmpy = y + dy[j]
                
                # Boundary check
                if tmpx < 0 or tmpx >= h or tmpy < 0 or tmpy >= w:
                    continue
                
                # Core logic:
                # 1. Point is within current kernel (curr_kernel[tmpx, tmpy] > 0)
                # 2. Point is not yet classified (pred[tmpx, tmpy] == 0)
                if curr_kernel[tmpx, tmpy] > 0 and pred[tmpx, tmpy] == 0:
                    pred[tmpx, tmpy] = l
                    # Push to queue
                    queue_arr[queue_tail, 0] = tmpx
                    queue_arr[queue_tail, 1] = tmpy
                    queue_arr[queue_tail, 2] = l
                    queue_tail += 1
        
        # Current layer processing complete
        # New points added (after current_layer_end) become seeds for next kernel layer expansion
        # queue_head doesn't need reset, continue processing
        
    return pred

class AnglePredictor:
    """Document page angle predictor using a deep learning model."""

    def __init__(self, device, cfg_path=None, ckpt_path=None):
        # Image normalization parameters (RGB channel mean & std)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.input_size = (1920, 1280)       # Default model input size (height, width)
        self.kernel_num = 4                  # Number of kernels for PSE
        self.min_area = 5                    # Minimum area threshold for connected component
        self.min_score = 0.5                 # Minimum score threshold for bounding box
        self.step = 13                       # Dynamic size adaptation steps
        self.step_size = 32                  # Step stride for dynamic adaptation
        self.use_adapt_size = True           # Enable adaptive input size

        # Check for checkpoint existence
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Model checkpoint not found: {ckpt_path}')

        # Hardcoded model configuration (from config.yaml)
        model_cfg = {
            'backbone': {
                'type': 'QARepVGG_B0_SLIM',
                'params': {
                    'pretrained': False,
                    'weights_path': 'weights',
                    'use_se': False
                }
            },
            'neck': {
                'type': 'REP_FPN',
                'params': {
                    'input_channels': [48, 96, 192, 384],
                    'planes': 48,
                    'block': 'QAREP'
                }
            },
            'feats_fuse': {
                'type': 'REP_Hourglass',
                'params': {
                    'inplanes': 192,
                    'inner_planes': 48,
                    'block': 'QAREP'
                }
            },
            'heads': [
                {
                    'type': 'UP_Head',
                    'name': 'kernels',
                    'params': {
                        'ch': 48,
                        'n_classes': 4,
                        'up_sample_times': 1
                    }
                },
                {
                    'type': 'UP_Head',
                    'name': 'angle_vec',
                    'params': {
                        'ch': 48,
                        'n_classes': 2,
                        'up_sample_times': 1
                    }
                },
                {
                    'type': 'UP_Head',
                    'name': 'thresh',
                    'params': {
                        'ch': 48,
                        'n_classes': 1,
                        'up_sample_times': 1
                    }
                }
            ],
            'trace_head_names': ['kernels', 'angle_vec']
        }

        # Load model configuration, weights and set device
        try:
            self.model = PTD(**model_cfg)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            self.model.load_state_dict(ckpt['state_dict'])
            self.model.set_test()  # Set model to test/eval mode if required
            self.model.eval()
            self.model = self.model.to(device).half()
            self.device = device
            if getattr(self.model, 'trace_channel_last', False):
                self.model = self.model.to(memory_format=torch.channels_last)
        except Exception as e:
            print("Exception log:", e)
            raise RuntimeError(f"Failed to load model: {e}")

        print("[INFO] AnglePredictor initialized.")

    def get_dynamic_size(self, image, set_size, step, step_size):
        """
        Compute adaptive input size based on image shape.
        Args:
            image: Input image (numpy array)
            set_size: Default size tuple (height, width)
            step: Number of discrete sizes
            step_size: Increment size for adaptation
        Returns:
            Adapted size (height, width) tuple
        """
        intervals = [[set_size[0] + i * step_size - step // 2 * step_size,
                      set_size[1] + i * step_size - step // 2 * step_size]
                     for i in range(step)]
        h, w = image.shape[:2]
        for interval in intervals:
            if w > interval[0] or h > interval[1]:
                return interval
        return intervals[-1]

    def pre_process(self, image, set_size):
        """
        Resize, pad, and normalize input image for network.
        Args:
            image: Original image (numpy array)
            set_size: Target input size
        Returns:
            Dict containing processed tensor and params.
        """
        h, w = image.shape[:2]
        im_scale = 1.0
        extreme_ratio = 2.5
        is_extreme = h > w * extreme_ratio or w > h * extreme_ratio
        is_long = max(h, w) > set_size[0]

        input_size = (
            self.get_dynamic_size(image, set_size, self.step, self.step_size)
            if self.use_adapt_size
            else set_size
        )

        # Compute scaling factor to ensure aspect ratio & size
        if is_extreme and is_long:
            area = h * w
            im_scale = np.sqrt(input_size[0] * input_size[1] * 3 / area)
        elif h > w:
            im_scale = input_size[0] / h if input_size[0] / h * w <= input_size[1] else input_size[1] / w
        else:
            im_scale = input_size[0] / w if input_size[0] / w * h <= input_size[1] else input_size[1] / h

        # Resize and pad to multiples of 64 for model compatibility
        new_size = (int(w * im_scale), int(h * im_scale))
        scaled = cv2.resize(image, dsize=new_size, interpolation=cv2.INTER_LINEAR)
        pad_r = 64 - (new_size[0] % 64)
        pad_b = 64 - (new_size[1] % 64)
        padded = cv2.copyMakeBorder(scaled, 0, pad_b, 0, pad_r, borderType=cv2.BORDER_CONSTANT, value=(128, 128, 128))

        # Convert to tensor and normalize
        tensor = transforms.ToTensor()(padded)
        tensor = transforms.Normalize(mean=self.mean, std=self.std)(tensor)
        return {
            "orig_image": image,
            "image": tensor.unsqueeze(0),
            "im_scale": im_scale
        }

    def forward_net(self, data):
        """
        Run model inference for processed input data.
        Args:
            data: Dict with processed image tensor
        Returns:
            Data dict with network output appended
        """
        with torch.no_grad():
            out = self.model(data["image"].to(self.device).half())
        data["net_out"] = out.float().cpu().numpy()
        return data
    
    def pse_4(self, kernals, min_area):
        """
        Progressive Scale Expansion (PSE) algorithm for text segmentation.
        
        Args:
            kernals: List of kernel arrays from large to small
            min_area: Minimum area threshold for connected components
            
        Returns:
            Segmentation prediction array with labeled regions
        """
        kernal_num = len(kernals)
        min_kernel = kernals[kernal_num - 1]

        # 1. Connected component analysis (keep using OpenCV as it's already fast)
        label_num, label, stats, centroids = cv2.connectedComponentsWithStats(min_kernel, connectivity=4)
        
        # Filter by area threshold
        small_areas_indices = np.where(stats[:, 4] < min_area)[0]
        if len(small_areas_indices) > 0:
            mask = np.isin(label, small_areas_indices)
            label[mask] = 0

        # 2. Prepare data for Numba processing
        # Convert list of arrays to 3D numpy array for efficient Numba handling
        # kernals order: [large, medium, small...] -> code logic expands from small->large
        # Original logic: kernals[kernal_num-1] is the smallest
        # Loop runs from kernal_num - 2 to 0
        kernals_arr = np.array(kernals)  # Shape: (K, H, W)
        
        # 3. Initialize queue
        # Find all initial seed points
        # Optimization: Only put edge points of minimum kernel?
        # Original logic puts all points with label>0, but this is wasteful
        # Actually only need points that have "background neighbors"
        # For simplicity and consistency, we first find all points
        # For Numba, we need to pre-allocate a sufficiently large array as queue
        h, w = label.shape
        # Maximum possible queue length is total image pixels
        queue_arr = np.zeros((h * w, 3), dtype=np.int32)
        
        # Get initial points (points with label > 0)
        # Use numpy for fast coordinate extraction
        points = np.column_stack(np.where(label > 0))
        current_q_len = points.shape[0]
        
        # Fill initial queue
        queue_arr[:current_q_len, 0] = points[:, 0]
        queue_arr[:current_q_len, 1] = points[:, 1]
        queue_arr[:current_q_len, 2] = label[points[:, 0], points[:, 1]]
        
        queue_head = 0
        queue_tail = current_q_len

        # 4. Call Numba function
        pred = label.copy()
        pred = pse_bfs_numba(pred, kernals_arr, queue_arr, queue_head, queue_tail)

        return pred
    
    def post_process(self, data):
        """
        Extract bounding boxes and vectors from model output maps.
        Args:
            data: Dict with net_out, im_scale
        Returns:
            Data dict updated with bboxes and vectors
        """
        net_out = data["net_out"]

        kernels = net_out[0, 0:self.kernel_num, :, :]
        kernels = [x.astype(np.uint8) for x in kernels]
        vec = net_out[0, self.kernel_num:self.kernel_num+2, :, :]
        score = net_out[0, -1, :, :]

        # Segment regions
        label = self.pse_4(kernels, self.min_area)
        label_num = np.max(label) + 1

        coords_all = np.stack(np.where(label > 0), axis=1)  # (y, x)
        labels_flat = label[label > 0]
        label_indices = [np.where(labels_flat == i)[0] for i in range(1, label_num)]
        points_per_label = [coords_all[idx] for idx in label_indices]

        bboxes, vectors = [], []

        def region_props(points):
            """Extract properties from a region defined by points."""
            if points.shape[0] < self.min_area:
                return None, None
            yx = points
            vec_x = np.mean(vec[0][yx[:, 0], yx[:, 1]])
            vec_y = np.mean(vec[1][yx[:, 0], yx[:, 1]])
            score_i = np.mean(score[yx[:, 0], yx[:, 1]])
            if score_i < self.min_score:
                return None, None
            rect = cv2.minAreaRect(yx[:, ::-1].astype(np.float32))  # (x, y)
            bbox = (cv2.boxPoints(rect) * 2) * (1.0 / data["im_scale"])
            bbox = bbox.astype('int32')
            return bbox, np.array([vec_x, vec_y])

        # Collect bounding boxes and vector directions for each region
        for points in points_per_label:
            bbox, vector = region_props(points)
            if bbox is not None:
                bboxes.append(bbox)
                vectors.append(vector)

        data["bboxes"] = bboxes
        data["vectors"] = vectors

        return data

    @staticmethod
    def norm_vec(vector):
        """Normalize given vector to unit length."""
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 1e-9 else np.zeros(2)

    @staticmethod
    def vec_to_angle(vector):
        """
        Convert 2D direction vector to angle in degrees [0, 360).
        (Horizontal right is 0)
        """
        ang = math.degrees(math.atan2(-vector[1], -vector[0]))
        return (ang + 360) % 360

    @staticmethod
    def box_center(p1, p2):
        """Calculate center point between two points."""
        return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

    def cal_page_angle(self, detect_res):
        """
        Compute best page orientation based on predicted regions.
        Args:
            detect_res: postprocessed dict with bboxes and vectors
        Returns:
            Estimated angle in degrees (float)
        """
        boxes = []
        for bbox, vector in zip(detect_res["bboxes"], detect_res["vectors"]):
            vector = self.norm_vec(vector)
            angle = self.vec_to_angle(vector)
            p1, p2, p3, p4 = bbox
            len1 = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
            len2 = np.hypot(p2[0] - p3[0], p2[1] - p3[1])
            if min(len1, len2) < 1e-6:
                continue
            # Select main axis by box aspect
            if len1 > len2:
                c1 = self.box_center(p1, p4)
                c2 = self.box_center(p2, p3)
            else:
                c1 = self.box_center(p1, p2)
                c2 = self.box_center(p3, p4)
            vec_wh = (c2[0] - c1[0], c2[1] - c1[1])
            angle_wh = self.vec_to_angle(vec_wh)
            angle_diff = abs((angle_wh - angle) % 180)
            # Filter out regions where detected vector doesn't match the box aspect axis
            if 45 < angle_diff < 135:
                continue
            wh_ratio = max(len1, len2) / min(len1, len2)
            boxes.append({"bbox": bbox, "angle": angle, "wh_ratio": wh_ratio})

        # Sort and select top candidates
        selected = sorted(boxes, key=lambda x: x["wh_ratio"], reverse=True)[:10]
        if not selected:
            return 0.0

        # Cluster similar angles
        selected = sorted(selected, key=lambda x: x["angle"])
        clusters, cur = [], []
        for box in selected:
            if not cur or abs(box["angle"] - cur[0]["angle"]) < 10:
                cur.append(box)
            else:
                clusters.append(cur)
                cur = [box]
        if cur:
            clusters.append(cur)
        # Handle wrapping angles across 360° boundaries
        if len(clusters) > 1:
            diff = (clusters[0][0]["angle"] - clusters[-1][0]["angle"] + 360) % 360
            if diff < 10:
                for box in clusters[-1]:
                    box["angle"] -= 360
                clusters[0].extend(clusters[-1])
                clusters = clusters[:-1]

        largest = max(clusters, key=len)
        angles = [box["angle"] for box in largest]
        page_angle = sum(angles) / len(angles)
        return page_angle % 360.0

    def predict(self, image_cv, verbose=False):
        """
        Main prediction interface: returns estimated angle in degrees for input image.
        Args:
            image_cv: OpenCV image (numpy array)
            verbose: Print timing and debug info
        Returns:
            Estimated page angle (float, degrees)
        """
        t0 = time.perf_counter()
        data = self.pre_process(image_cv, self.input_size)
        t1 = time.perf_counter()
        if verbose: print(f"[Preprocess]: {(t1 - t0) * 1000:.2f} ms")
        data = self.forward_net(data)
        t2 = time.perf_counter()
        if verbose: print(f"[Inference]: {(t2 - t1) * 1000:.2f} ms")
        data = self.post_process(data)

        angle = self.cal_page_angle(data)
        t3 = time.perf_counter()
        if verbose: print(f"[Postprocess]: {(t3 - t2) * 1000:.2f} ms")
        print(f"[INFO] Estimated angle: {angle:.2f}°")
        return angle
