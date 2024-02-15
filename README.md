# eventshow
(inspired by [imshow](https://github.com/wkentaro/imshow), creating an event2frame representation software for commandline usage)

## Two extensible sub-packages:
- **readwrite**: for different datasets, there will be different event loaders and writers.
- **event2frame**: implementation for different event representation methods. Currently:
    - **SBT**
      <img src="https://github.com/RunqiuBao/eventshow/blob/main/.readme/sbt.png" width="200", height="150">
    - **SBN**
      <img src="https://github.com/RunqiuBao/eventshow/blob/main/.readme/sbn.png" width="200", height="150">
    - **Concentrated SBN**: from [se-cff](https://github.com/yonseivnl/se-cff).
      <img src="https://github.com/RunqiuBao/eventshow/blob/main/.readme/concentrate.png" width="200", height="150">

## Example:
- for dsec dataset, use SBT, save 10 sample frames:
```
eventshow /home/runqiu/Desktop/dsec/interlaken_00_a/events/left/events.h5 -o /home/runqiu/Desktop/dsec/interlaken_00_a/events/left/ --dtms 20 --numframes 10  --rw_module dsec
```