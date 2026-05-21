import os
import cv2
import numpy as np
from PIL import Image
import urllib.request
import ssl
import sys

from .angle_predictor.angle_predictor import AnglePredictor

class AngleCorrector(object):
    """
    AngleCorrector is used to predict and correct the rotation angle of an image.
    Supports both OpenCV and PIL image input.
    """

    def __init__(self, model_path, pad_value=(255, 255, 255)):
        """
        Args:
            pad_value (tuple): Padding value for rotated image border.
        """
        # Use model_weight folder in the same directory as this script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if not model_path:
            self.model_path = os.path.join(current_dir, "model_weight")
        else:
            self.model_path = model_path

        self.model = None
        self.pad_value = pad_value
        self.initialize_model()

    def _download_model(self, ckpt_path):
        """
        Download model file from remote repository if not exists or is a Git LFS pointer.
        
        Args:
            ckpt_path (str): Path to the model checkpoint file.
        
        Returns:
            bool: True if download successful, False otherwise.
        """
        model_url = "https://github.com/TencentCloudADP/youtu-parsing/releases/download/v1.0.0/model.pth"
        
        try:
            print(f"Downloading model from {model_url}...")
            print("This may take a few minutes (model size: ~106MB)...")
            
            # Create directory if not exists
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            
            # Download with progress (bypass SSL verification for environments without root certs)
            def reporthook(count, block_size, total_size):
                if total_size > 0:
                    percent = min(int(count * block_size * 100 / total_size), 100)
                    sys.stdout.write(f"\rDownloading: {percent}%")
                    sys.stdout.flush()
            
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ssl_context)
            )
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(model_url, ckpt_path, reporthook=reporthook)
            print("\n✓ Model downloaded successfully!")
            return True
            
        except Exception as e:
            print(f"\n✗ Failed to download model: {e}")
            return False

    def initialize_model(self):
        """
        Load the angle prediction model from the given path.
        Auto-download if model not found or is a Git LFS pointer.
        """
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path, exist_ok=True)
        
        ckpt_path = os.path.join(self.model_path, "model.pth")
        
        # Check if model exists and is valid (not a LFS pointer)
        needs_download = False
        if not os.path.exists(ckpt_path):
            print(f"Model checkpoint not found at: {ckpt_path}")
            needs_download = True

        elif os.path.getsize(ckpt_path) < 1024 * 1024:  # Less than 1MB, likely invalid
            print(f"Model file seems incomplete (size: {os.path.getsize(ckpt_path)} bytes)")
            needs_download = True
        
        # Download if needed
        if needs_download:
            print("Attempting to download model automatically...")
            if not self._download_model(ckpt_path):
                raise FileNotFoundError(
                    f"Model checkpoint not found or invalid at: {ckpt_path}\n"
                    f"Please manually download from:\n"
                    f"https://github.com/TencentCloudADP/youtu-parsing/releases/download/v1.0.0/model.pth"
                )
        
        print(f"Loading angle correction model from: {self.model_path}")
        # No longer need cfg_path, config is hardcoded in AnglePredictor
        self.model = AnglePredictor("cuda", ckpt_path=ckpt_path)

    def rotate_image(self, image: np.ndarray, angle: float, pad_value=(0, 0, 0)):
        """
        Rotate the image by a specific angle with padding.

        Args:
            image (np.ndarray): The image to rotate.
            angle (float): The rotation angle in degrees. Positive values mean counter-clockwise rotation.
            pad_value (tuple): Padding value for border.

        Returns:
            rotated (np.ndarray): Rotated image.
            M (np.ndarray): Rotation affine matrix (2x3).
        """
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Obtain the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Compute new bounding dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust rotation matrix for translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Apply affine warp with border padding
        rotated = cv2.warpAffine(
            image, M, (new_w, new_h),
            borderMode=cv2.BORDER_CONSTANT, borderValue=pad_value
        )
        return rotated, M

    def pillow_to_opencv(self, img: Image.Image) -> np.ndarray:
        """
        Convert a PIL Image to an OpenCV numpy array.

        Args:
            img (PIL Image): Image to convert.

        Returns:
            np.ndarray: Converted image in OpenCV format.
        """
        np_img = np.array(img)
        if img.mode == 'RGB':
            return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        elif img.mode == 'RGBA':
            return cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGRA)
        elif img.mode == 'L':
            return np_img  # Grayscale
        else:
            raise ValueError(f'Unsupported image mode: {img.mode}')

    def opencv_to_pillow(self, np_img: np.ndarray) -> Image.Image:
        """
        Convert an OpenCV numpy array image to PIL Image.

        Args:
            np_img (np.ndarray): OpenCV format image.

        Returns:
            PIL Image: Converted PIL image.
        """
        if np_img.ndim == 2:
            return Image.fromarray(np_img)  # Grayscale
        elif np_img.shape[2] == 3:
            return Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
        elif np_img.shape[2] == 4:
            return Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGRA2RGBA))
        else:
            raise ValueError(f'Unsupported image shape: {np_img.shape}')

    def transform_points_inverse(self, points: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Apply inverse affine transformation to a set of 2D points.

        Args:
            points (np.ndarray): Shape (N, 2), coordinate points.
            M (np.ndarray): Affine matrix of shape (2, 3).

        Returns:
            np.ndarray: Transformed coordinates of shape (N, 2).
        """
        M_inv = cv2.invertAffineTransform(M)  # Compute inverse affine (2x3)
        ones = np.ones((points.shape[0], 1), dtype=np.float32)   # (N, 1)
        pts_homo = np.hstack([points, ones])                     # (N, 3)
        orig_pts = np.dot(pts_homo, M_inv.T)                     # (N, 2)
        return orig_pts

    def coord_inverse_rotation(self, coord, M):
        """
        Map rotated rectangle coordinates back to the original image.

        Args:
            coord (list): [x1, y1, ..., x4, y4], coordinates after rotation.
            M (np.ndarray): Affine transformation matrix (2x3).

        Returns:
            list: Original image coordinates [x1, y1, ..., x4, y4].
        """
        if M is None:
            return [int(x) for x in coord]
        # Extract corner points
        corners = [(coord[j], coord[j+1]) for j in range(0, 8, 2)]
        corners = np.array(corners, dtype=np.float32)   # (4, 2)
        # Apply inverse affine transformation
        orig_corners = self.transform_points_inverse(corners, M)   # (4, 2)
        # Flatten and round to int
        poly = orig_corners.reshape(-1).tolist()
        poly = [int(round(x)) for x in poly]
        return poly

    def __call__(self, image):
        """
        Correct the angle of the input image.

        Args:
            image (np.ndarray or PIL.Image): Input image.

        Returns:
            image_cv_rotated (same type as input): Rotated image.
            M (np.ndarray): Affine transformation matrix used for rotation (2x3).
        """
        is_cv_image = not isinstance(image, Image.Image)
        if not is_cv_image:
            image_cv = self.pillow_to_opencv(image)
        else:
            image_cv = image

        # Predict page angle
        page_angle = self.model.predict(image_cv, verbose=True)

        if page_angle < 3 or page_angle > 357:
            page_angle = 0

        print(f"Final angle: {page_angle:.2f}")

        # Rotate image to correct orientation
        image_cv_rotated, M = self.rotate_image(image_cv, -page_angle, pad_value=self.pad_value)

        # Convert back to PIL if input was PIL
        if not is_cv_image:
            image_cv_rotated = self.opencv_to_pillow(image_cv_rotated)

        return image_cv_rotated, M, page_angle
        