# VI-Net For Orientation Estimation.

Modified from the official implementation of [VI-Net](https://github.com/JiehongLin/VI-Net)

## Requirements
The code has been tested with:
- Python 3.10.19
- PyTorch 2.0.0
- CUDA 11.8.0

Other dependencies:

```bash
sh dependencies.sh
```
## Data

For testing, please download the test dataset from [here](https://huggingface.co/datasets/kongbai1aa/canoverse/blob/main/canoverse_test.zip) and place it in the `dataset` directory.


## Network Training


Train VI-Net for rotation estimation:

```
python vinet_train.py
```

## Evaluation

To test the model, please run:

```
python vinet_test.py
```
