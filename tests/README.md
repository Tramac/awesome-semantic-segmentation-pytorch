# Overfitting Test

In order to ensure the correctness of models, the project provides a overfitting test (a trick which makes the train set and the val set includes the same images) script.

### Usage

<img src='./test_img.jpg' width = '100' div align=right /> <img src = './test_mask.png' width = '100' height = '100' div align = right />

​                                           (a) img: 2007_000033.jpg                                                                                                     (b) mask: 2007_000033.png

### Test Result

| Model  | epoch | mIoU  | pixAcc |
| :----- | ----- | ----- | ------ |
| FCN32s | 300   | 94.0% | 98.2%  |

