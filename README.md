# Face-detection-with-key-point
This is a human face detection project, which can detect human faces and mark key points at the same time

这是一个人脸与人脸关键点检测的 demo，运行 mask 文件夹中的 ```detect.py``` 实现人脸检测。

核心代码来自于[libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train)，拥有很快的检测速度（FPS：~60）。

我在原先的基础上做了轻量与优化，使之能够批量处理图片并能处理视频。

