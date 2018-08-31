
# Mask CNN:

## Abstract
This is a tensorflow re-implementation of [Mask RCNN](http://cn.arxiv.org/abs/1703.06870).

The original implement will be available [here](https://github.com/matterport/Mask_RCNN), if the code is helpful, please cite the [paper](http://cn.arxiv.org/abs/1703.06870). For shortness and efficiency, we re-build the code structor, you can easilly annotate your dataset, and train or test your data based on this repository.


## Structor
### Code structor
```
.
├── build
    ├── bdist.win-amd64
    ├── lib
        ├── mrcnn
├── config
    ├── cfgs.py
├── datas
    ├── box
        ├── dataset.py
    ├── multi-box
        ├── dataset.py
├── dist
├── logs
├── mask_rcnn.egg-info
├── mrcnn
├── output
├── tools
    ├── inference.py
    ├── train.py
├── utils
    ├── utils.py
...
├── README.md
```


## Installation
### Install  tensoflow-gpu
We succeed in implementing this repository in keras=2.0.8, and tensorflow-gpu=1.6 on Nvidia 1080TI. Envirionments of other versions may be a little different. For other details, please refer to the original [code](https://github.com/matterport/Mask_RCNN).

### Download Model
1. You can download the pre-trained weights [here](https://pan.baidu.com/s/1Q_4GqQu4tN0WhOUc2tZFtg), and place it in the root directory of the project.

## Training and testing
> 1. **Data Preparing:** Use [labelme](https://github.com/wkentaro/labelme) to get annotations for your data, for our final goal, we just need to provide three directories for training, **rgb_train** (for image directory), **mask** (for mask png directory) and **yaml** (for yaml directory). For some data processing and useful tools reference, you can refer [here](https://github.com/gzp0201/ObjectDetectionTools), if you think it is useful for you, please star it without hesitatation.
> 2. For short, you need to make a directory under **./datas** named as your data's name like **box** and create a python script named **dataset.py**, in which you should edit the parameter **label_name_dict**.
> 3. **./config/cfg.py** is necessary to be modified to satisfied your situation.
> 4. After finishing the preparation, do some change in **train.py**, and then just run **python train.py**, then the training process will start.
> 5. If you want to inference your test data, make some changes in your **inference.py** and run **python inference.py** for inference after setting the image directory when training process terminates.

### Example for box -surface detection

- To annotate the data, I just use the [labelme](https://github.com/wkentaro/labelme) to get my dataset.
- After the annotation, data format converting will be inplemented, our goal is to generate the three directories as described before,.
- Re-edit the *cfgs.py* and *label_dict.py*.
- Training.
- Edit the image directory for inference and test data with script *inference.py*

### Experiments show

#### Good results:

| ![]() | ![]() |
| ----- | ----- |
| ![]() | ![]() |
| ![]() | ![]() |



#### Bad results:

| ![]() | ![]() |
| ----- | ----- |
| ![]() | ![]() |
| ![]() | ![]() |

