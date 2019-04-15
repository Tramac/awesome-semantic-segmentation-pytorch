# Overfitting Test

In order to ensure the correctness of models, the project provides a overfitting test (a trick which makes the train set and the val set includes the same images) script.

### Usage

<img src='./test_img.jpg' width = '300' height = '220' /> <img src = './test_mask.png' width = '300' height = '220' />

　　　(a) img: 2007_000033.jpg  　　　　　　　(b) mask: 2007_000033.png


### Test Result

| Model  | epoch | mIoU  | pixAcc |
| :-----: | :-----: | :-----: | :------: |
| FCN32s | 300   | 94.0% | 98.2%  |

