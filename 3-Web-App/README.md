# 用ML模型来构建一个web 应用程序（本篇是我自己翻译的^_^）

在本节课程中，将介绍如何将你的Scikit-learn模型保存为文件，以便在web应用程序中进行预测。 保存模型后，你将学习如何在Flask中构建的web应用程序中使用它。 首先使用UFO目击数据创建一个模型， 然后，构建一个web应用程序，你输入带有纬度和经度值的秒数，以预测哪个国家报告看到了不明飞行物。  

![UFO 目击数据](images/ufo.jpg)

图片提供 <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michael Herren</a> 来自<a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  

## 课程

1. [创建一个 Web 应用](1-Web-App/README.md)

## 致谢

"创建一个 Web 应用" 作者是 [Jen Looper](https://twitter.com/jenlooper)。

“小测试作者”是 Rohan Raj.

数据集来源 [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings)。

web应用的框架来自Abhinav Sagar 的部分建议，分别为[这篇文章](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) and [代码库](https://github.com/abhinavsagar/machine-learning-deployment) 。