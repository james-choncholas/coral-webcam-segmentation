# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import svgwrite
import re
import PIL
import argparse
from functools import partial
from collections import deque

import numpy as np
import scipy.ndimage
import scipy.misc
from PIL import Image

import gstreamer
#from pose_engine import PoseEngine, EDGES, BODYPIX_PARTS

from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter

# Color mapping for bodyparts
#RED_BODYPARTS = [k for k,v in BODYPIX_PARTS.items() if "right" in v]
#GREEN_BODYPARTS = [k for k,v in BODYPIX_PARTS.items() if "hand" in v or "torso" in v]
#BLUE_BODYPARTS = [k for k,v in BODYPIX_PARTS.items() if "leg" in v or "arm" in v or "face" in v or "hand" in v]

#def shadow_text(dwg, x, y, text, font_size=16):
#    dwg.add(dwg.text(text, insert=(x + 1, y + 1), fill='black',
#                     font_size=font_size, style='font-family:sans-serif'))
#    dwg.add(dwg.text(text, insert=(x, y), fill='white',
#                     font_size=font_size, style='font-family:sans-serif'))
#
#def draw_pose(dwg, pose, color='blue', threshold=0.2):
#    xys = {}
#    for label, keypoint in pose.keypoints.items():
#        if keypoint.score < threshold: continue
#        xys[label] = (int(keypoint.yx[1]), int(keypoint.yx[0]))
#        dwg.add(dwg.circle(center=(int(keypoint.yx[1]), int(keypoint.yx[0])), r=5,
#                           fill='cyan', stroke=color))
#    for a, b in EDGES:
#        if a not in xys or b not in xys: continue
#        ax, ay = xys[a]
#        bx, by = xys[b]
#        dwg.add(dwg.line(start=(ax, ay), end=(bx, by), stroke=color, stroke_width=2))


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.
  Returns:
    A Colormap for visualizing segmentation results.
  """
  #colormap = np.zeros((256, 3), dtype=int)
  #indices = np.arange(256, dtype=int)

  #for shift in reversed(range(8)):
  #  for channel in range(3):
  #    colormap[:, channel] |= ((indices >> channel) & 1) << shift
  #  indices >>= 3



  #15 is person, 20 is wall?
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[15] = (255,255,255)
  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.
  Args:
    label: A 2D array with integer type, storing the segmentation label.
  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.
  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

class Callback:
  ##def __init__(self, engine, anonymize=True, bodyparts=True):
  def __init__(self, interpreter):
    ##self.engine = engine
    self.interpreter = interpreter
    ##self.anonymize = anonymize
    ##self.bodyparts = bodyparts
    self.background_image = None
    self.last_time = time.monotonic()
    self.frames = 0
    self.sum_fps = 0
    self.sum_process_time = 0
    #self.sum_inference_time = 0

  def __call__(self, image):
    #i = Image.frombytes('RGB', (image.shape[1], image.shape[0]), image.copy())#, "raw", 'RGB', stride) # this works
    #i.save("/tmp/wtf.jpg")

    common.set_input(self.interpreter, np.ascontiguousarray(image))
    self.interpreter.invoke()
    result = segment.get_output(self.interpreter)
    if len(result.shape) == 3:
      result = np.argmax(result, axis=-1)

    ##mask = Image.fromarray(label_to_color_image(result).astype(np.uint8))
    mask = label_to_color_image(result)
    #output_image = mask

    bg = np.full(image.shape, [0, 255, 0], dtype=np.uint8)
    #output_image = bg

    # use mask to combine with background
    tmp1 = np.bitwise_and(image, mask)
    tmp2 = np.bitwise_and(bg, ~mask)
    output_image = tmp1+tmp2

    end_time = time.monotonic()

    self.frames += 1
    self.sum_fps += 1.0 / (end_time - self.last_time)
    ##self.sum_process_time += 1000 * (end_time - start_time) - inference_time
    ##self.sum_inference_time += inference_time
    self.last_time = end_time
    text_line = 'PoseNet: %.1fms Frame IO: %.2fms TrueFPS: %.2f Nposes %d' % (
        ##self.sum_inference_time / self.frames,
        0,
        ##self.sum_process_time / self.frames,
        0,
        self.sum_fps / self.frames,
        #len(poses)
        0
    )

    print(text_line)
    return output_image

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mirror', help='flip video horizontally', action='store_true')
    parser.add_argument('--model', help='.tflite model path.', required=False)
    parser.add_argument('--width', help='Source width', default='640')
    parser.add_argument('--height', help='Source height', default='480')
    parser.add_argument('--videosrc', help='Which video source to use', default='/dev/video0')

    parser.add_argument('--anonymize', dest='anonymize', action='store_true', help='Use anonymizer mode [--noanonymize]')
    parser.add_argument('--noanonymize', dest='anonymize', action='store_false', help=argparse.SUPPRESS)
    parser.set_defaults(anonymize=False)

    parser.add_argument('--bodyparts', dest='bodyparts', action='store_true', help='Color by bodyparts [--nobodyparts]')
    parser.add_argument('--nobodyparts', dest='bodyparts', action='store_false', help=argparse.SUPPRESS)
    parser.set_defaults(bodyparts=True)

    parser.add_argument('--h264', help='Use video/x-h264 input', action='store_true')
    parser.add_argument('--jpeg', help='Use video/jpeg input', action='store_true')
    args = parser.parse_args()

    if args.h264 and args.jpeg:
        print('Error: both mutually exclusive options h264 and jpeg set')
        sys.exit(1)

    #default_model = 'models/bodypix_mobilenet_v1_075_640_480_16_quant_edgetpu_decoder.tflite'
    #default_model = 'models/deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite'
    default_model = 'models/deeplabv3_mnv2_pascal_quant_edgetpu.tflite'
    model = args.model if args.model else default_model
    print('Model: {}'.format(model))


    ##engine = PoseEngine(model)
    ##inference_size = (engine.image_width, engine.image_height)
    interpreter = make_interpreter(model, device=':0')
    interpreter.allocate_tensors()
    inference_size = common.input_size(interpreter)
    #inference_size = [512,512]
    print('Inference size: {}'.format(inference_size))

    src_size = (int(args.width), int(args.height))
    if args.videosrc.startswith('/dev/video'):
        print('Source size: {}'.format(src_size))

    gstreamer.run_pipeline(##Callback(engine,
                           ##         anonymize=args.anonymize,
                           ##         bodyparts=args.bodyparts),
                           Callback(interpreter),
                           src_size, inference_size,
                           mirror=args.mirror,
                           videosrc=args.videosrc,
                           h264=args.h264,
                           jpeg=args.jpeg)


if __name__ == '__main__':
    main()
