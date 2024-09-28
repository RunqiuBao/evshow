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
- **readwrite**: for different datasets, there will be different event data loaders.
    - **dsec** 
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

## Install and Usage:
- **Install**:
  - Build from source:
    ```bash
    git clone https://github.com/RunqiuBao/evshow.git
    pip install -r requirements.txt
    python3 -m pip install ./
    ```
  - Pip:
- **Example usage**:
  - On [dsec](https://github.com/uzh-rpg/DSEC) format data:
    ```bash
    cd ./evshow
    
    # install git LFS if not yet
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install git-lfs
    git lfs install
    # use git LFS to download sample data
    git lfs pull
    
    # accumulating events every 200000, using e2vid conversion, generate first 10 frames.
    evshow data/sample_dsecformat.h5 -o data/seq0/ --numevents 200000 --rw_module dsec --e2vid --numframes 10
    ```
