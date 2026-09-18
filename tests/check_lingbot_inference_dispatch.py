"""Dispatch the declared official mode, including window overlap and CPU offload."""
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'modal_apps'))
from lingbot_room import infer_sequence
calls=[]
model=SimpleNamespace(inference_streaming=lambda images,**kw:calls.append(('streaming',images,kw)),inference_windowed=lambda images,**kw:calls.append(('windowed',images,kw)))
config={'anchor_frames':8,'keyframe_interval':1}
infer_sequence(model,'images',config,'cpu')
assert calls[-1]==('streaming','images',{'num_scale_frames':8,'keyframe_interval':1,'output_device':'cpu'})
w={**config,'mode':'windowed','window_size':64,'overlap_size':16}
infer_sequence(model,'images',w,'cpu')
assert calls[-1]==('windowed','images',{'num_scale_frames':8,'keyframe_interval':1,'output_device':'cpu','window_size':64,'overlap_size':16,'overlap_keyframes':None})
for bad in [{**w,'mode':'unknown'},{**w,'overlap_size':64},{**w,'window_size':0}]:
 before=len(calls)
 try:infer_sequence(model,'images',bad,'cpu')
 except ValueError:pass
 else:raise AssertionError('Invalid mode/window accepted')
 assert len(calls)==before
print('PASS: exact official streaming/windowed dispatch and overlap bounds')
