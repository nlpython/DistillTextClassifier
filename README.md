基于BERT->BiLSTM的蒸馏实验
================

参考论文《Distilling Task-Specific Knowledge from BERT into Simple Neural Networks》



 - student模型(BiLSTM) 准确率在 0.80 ~ 0.82

 - teacher(BERT) 准确率在 0.91 ~ 0.93

 - 蒸馏模型 准确率在 0.84 ~ 0.86



## 使用方法

1. 将`bert-base-chinese`下载到本地
2. 运行`python train_teacher.py`训练teacher模型，并保存至`checkpoints/`
3. 运行`python distil.py`进行蒸馏，并保存至`checkpoints/`

训练日志保存至`logs/` <br>
可以通过调整`utils/hyperParams.py`中的参数来调整蒸馏模型的参数<br>
`tempurature`和`alpha`参数会显著影响蒸馏效果<br>
更多蒸馏`loss`有待更新

