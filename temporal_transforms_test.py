import random
import math

from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)
from temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)
from temporal_transforms import Compose as TemporalCompose

frame_indices = range(30)
print(frame_indices)
for i in range(10):
#    t = TemporalRandomCrop(16)
#    t = TemporalCenterCrop(16)
#    t = TemporalEvenCrop(16, 2)
#    t = SlidingWindow(16)
#    t = TemporalSubsampling(2)
#    t = LoopPadding(16)
    results = t(frame_indices)
    print(results)
