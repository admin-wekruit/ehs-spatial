"""Resolution changes must preserve the calibrated crop and native depth domain."""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'modal_apps'))
from droid_room import prepare_image, SOURCE_K, SOURCE_D

image = np.zeros((480, 640, 3), np.uint8)
image[70:360, 110:400] = [60, 150, 220]
calibration = {'source_K_fx_fy_cx_cy': SOURCE_K, 'source_distortion': SOURCE_D}
low, low_k = prepare_image(image, calibration)
high, high_k = prepare_image(image, calibration, resolution_scale=2)
fx, fy, cx, cy = SOURCE_K
rectified = cv2.undistort(image, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]), np.array(SOURCE_D))
assert np.array_equal(low, cv2.resize(rectified, (352, 256))[8:-8, 16:-16])
assert np.array_equal(high, cv2.resize(rectified, (704, 512))[16:-16, 32:-32])
assert high.shape == (480, 640, 3) and np.array_equal(high_k, low_k * 2)
for bad in [0, 3, 1.5, True]:
    try: prepare_image(image, calibration, resolution_scale=bad)
    except ValueError: pass
    else: raise AssertionError('Unbounded image resolution accepted')
try: prepare_image(image[:240], calibration)
except ValueError: pass
else: raise AssertionError('Source raster incompatible with calibration accepted')
print('PASS: declared resolution preserves calibrated crop, pixels and intrinsics')
