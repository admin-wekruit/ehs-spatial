"""RGB rectification and SAM masks must share the calibrated source domain."""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'modal_apps'))
from lingbot_cloud_compare import rectify_raster
k=np.array([[517.306408,0,318.64304],[0,516.469215,255.313989],[0,0,1.]])
c={'K':k.tolist(),'distortion':[.262383,-.953104,-.005358,.002628,1.163314],'source_wh':[640,480]}
y,x=np.indices((480,640)); rgb=np.stack((x%256,y%256,(x+y)%256),-1).astype('uint8')
actual=rectify_raster(rgb,c)
expected=cv2.undistort(rgb,k,np.array(c['distortion']),None,k)
assert np.array_equal(actual,expected)
mask=((x>90)&(x<230)&(y>140)&(y<350))
mx,my=cv2.initUndistortRectifyMap(k,np.array(c['distortion']),None,k,(640,480),cv2.CV_32FC1)
expected_mask=cv2.remap(mask.astype('uint8'),mx,my,cv2.INTER_NEAREST)
assert np.array_equal(rectify_raster(mask.astype('uint8'),c,mask=True),expected_mask)
assert np.array_equal(rectify_raster(rgb,None),rgb)
assert np.array_equal(rectify_raster(rgb,{**c,'distortion':[0]*5}),rgb)
for bad in [{**c,'source_wh':[320,240]}, {**c,'K':[[0]*3]*3}, {**c,'distortion':[float('nan')]*5}]:
 try:rectify_raster(rgb,bad)
 except ValueError:pass
 else:raise AssertionError('Invalid calibration accepted')
print('PASS: calibrated RGB and mask rectification share the same source raster')
