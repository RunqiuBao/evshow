<div align="center">
  <img src="https://github.com/RunqiuBao/evshow/blob/main/.readme/evshow.png" width="200", height="200">
  <h1>Evshow</h1>
  <p>
    <b>Convert event data into frame representation.</b>
  </p>
  <br>
  <br>
  <br>
</div>

Inspired by [imshow](https://github.com/wkentaro/imshow), creating an event2frame representation software for commandline usage

## Two extensible sub-packages:
- **readwrite**: for different datasets, there will be different event loaders and writers.
- **event2frame**: implementation for different event representation methods. Currently:
    - **SBT**
      <div align="left">
      <img src="https://github.com/RunqiuBao/evshow/blob/main/.readme/sbt.png" width="200", height="150">
      </div>
    - **SBN**
      <div align="left">
      <img src="https://github.com/RunqiuBao/evshow/blob/main/.readme/sbn.png" width="200", height="150">
      </div>
    - **Concentrated SBN**: from [se-cff](https://github.com/yonseivnl/se-cff).
      <div align="left">
      <img src="https://github.com/RunqiuBao/evshow/blob/main/.readme/concentrate.png" width="200", height="150">
      </div>
    - **rpg_e2vid**: from [e2vid](https://github.com/uzh-rpg/rpg_e2vid)).
      <div align="left">
      <img src="https://github.com/RunqiuBao/evshow/blob/main/.readme/e2vid.png" width="200", height="150">
      </div>

## Example:
- for dsec dataset, use SBT, save 10 sample frames:
```
eventshow /home/runqiu/Desktop/dsec/interlaken_00_a/events/left/events.h5 -o /home/runqiu/Desktop/dsec/interlaken_00_a/events/left/ --dtms 20 --numframes 10  --rw_module dsec
```
