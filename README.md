# eventshow
(inspired by (imshow)[https://github.com/wkentaro/imshow], creating an event2frame representation software for commandline usage)

## Two extensible sub-package:
- *readwrite*: for different datasets, there will be different event loaders and writers.
- *event2frame*: implementation for different event representation methods. Currently:
    - SBT
    - SBN

## Example:
- for dsec dataset, use SBT, save 10 sample frames:
```
eventshow /home/runqiu/Desktop/dsec/interlaken_00_a/events/left/events.h5 -o /home/runqiu/Desktop/dsec/interlaken_00_a/events/left/ --dtms 20 --numframes 10  --rw_module dsec
```