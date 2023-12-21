<h1 align="center">
益选课友:（这个标题真是糖的出奇啊！！！）
</h1>
<p align="center">
  2023 Fall BUSS 3602 人工智能导论大作业
  <br />
  李金昊; 卢其汶; 吴苛铭; 熊栩源; 赵霄宇
  <br />
</p>

这是一款可以帮助安泰同学排课的脚本😊

## Requirements

为了让程序正常运行，你需要安装：

* `python`
* `numpy`
* `PIL`
* `PyPt5`
* `pytorch`

你可以按照如下的操作进行：

* 创建一个虚拟环境并激活：

  ```
  conda create -n CourseAssistant python=3.11
  conda activte CourseAssistant
  ```

* 安装对应cuda版本的[pytorch](https://pytorch.org/)，例如：

  ```
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

* 安装`Pillow`，`Pandas`，`PyQt5`

  ```
  pip install Pillow
  pip install pandas
  pip install PyQt5
  ```

## Quickstart

采用如下命令会直接显示出UI窗口：

```
python runner.py
```

用户可以根据自己的需要选择偏好(如喜好的时间，老师等)：

XXX插入图片()

## Algorithm

核心算法采用了AC3+forward checking

## Update model

本项目的实现基于了一个默认模型，如果想要获得更好的用户体验，可以在运行程序的时候加上参数`is_update`：

```
python runner.py --is_update=True
```

