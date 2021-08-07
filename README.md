# 面向初学者的机器学习课程

> 🌍 通过世界文化探索机器学习，周游世界 🌍

这是一份有微软Azure云提供的机器学习课程，课程时长12周、包含24课时。在这份课程里，我们将避开深度学习的内容，使用Scikit-learn这个工具来学习所谓的“经典机器学习”算法。同样，我们还有一门兄弟课程“面向初学者的数据科学”。

就像环游世界一样，这门课程的数据采自全世界各地。同时，每一节课程都有一个课前和课后测试，包含了课程的介绍、具体内容以及课后作业等等。课程基于项目实践的方式进行教学，者被证明是一种有效的学习方式。

**✍️ 重点感谢我们的作者** Jen Looper, Stephen Howell, Francesca Lazzeri, Tomomi Imura, Cassie Breviu, Dmitry Soshnikov, Chris Noring, Ornella Altunyan, and Amy Boyd

**🎨 同样感谢我们的内容插画作者** Tomomi Imura, Dasani Madipalli, and Jen Looper

**🙏 同样感谢微软的学生代表作者，审核人以及内容贡献者 🙏 **, 尤其是 Rishit Dagli, Muhammad Sakib Khan Inan, Rohan Raj, Alexandru Petrescu, Abhishek Jaiswal, Nawrin Tabassum, Ioan Samuila, and Snigdha Agarwal

**🤩 特别感谢微软学生代表Eric Wanjau为我们提供的R课程!** 

---

# 从这里开始

**作为学生**, fork整个repo到自己的github账号里，自己活着和团队协作完成里边的练习:

- 完成课前测试。Start with a pre-lecture quiz.
- 阅读课程内容，并及时停下来思考、检测自己已经学习的内容。Read the lecture and complete the activities, pausing and reflecting at each knowledge check.
- 课程代码都在每个项目的'/solution'文件夹下，但是我们还是建议所有人在理解的基础上自己创建工程来实践，而不是直接运行现有的代码。Try to create the projects by comprehending the lessons rather than running the solution code; however that code is available in the `/solution` folders in each project-oriented lesson.
- 完成课后测试Take the post-lecture quiz.
- 完成挑战Complete the challenge.
- 完成作业Complete the assignment.
- 当完成课程后，访问 [Discussion board](https://github.com/microsoft/ML-For-Beginners/discussions)并通过PAT标注记录当前学习情况（此块不准确）。 我们通过填写PAT这个表格，来完成我们学习的评估。当然我们参与到其他的PAT中，从而可以和其他人一块学习。

> 进一步的学习，我们推荐您查看 [Microsoft Learn](https://docs.microsoft.com/en-us/users/jenlooper-2911/collections/k7o7tg1gp306q4?WT.mc_id=academic-15963-cxa) 的内容和学习路径

**作为老师**, 请查阅 [included some suggestions](for-teachers.md)。

---

## 开发团队

[![Promo video](ml-for-beginners.png)](https://youtu.be/Tj1XWrDSYJU "Promo video")

> 🎥 点击上面的图片查看关于这个项目和创建者的视频。

---

## 教学策略

在保证教程具有很深的聚焦性的基础上，教程还有两个重要的学习理念，分别为确保你真正的参与到项目实操以及频繁的测验。

通过确保内容与项目一致，这个过程对学生来说更有吸引力，概念的留存也会增强。 这个课程设计灵活、有趣，可以全部或部分学习。 项目开始时很小，在12周周期结束时变得越来越复杂。   

为了让学习更有吸引力，同时学完后概念能记得更深刻，课程的内容在设计上和项目是一致的。 此外，课前的小测验可以确定学生学习某一主题的意图，而课后的小测验则可以确保进一步的记忆。本课程还包括一个关于ML的实际应用的后记，可以学有余力的学生继续学习的基础。

> 可以在[规范](CODE_OF_CONDUCT.md)、[提交贡献](CONTRIBUTING.md)以及[参与翻译](TRANSLATIONS.md)这三个连接找到你对应感兴趣的内容，欢迎随时提供反馈。

## 每节课的包含内容:

- （可选）sketchnote
- （可选）视频
- 课前小测试
- 编写课程
- 对于基于项目的课程，将逐步指导如何构建项目
- 知识点检查
- 一个小挑战
- 补充阅读
- 作业
- 课后测试

> **关于小测试**: 所有的小测试都在这个[网站](https://jolly-sea-0a877260f.azurestaticapps.net)上, 共50个测试，每个测试包含三道题目。 这些测试的连接都包含在课程中，当然你也可以通过在本地文件夹"quiz-app"文件夹里操作说明在本地运行查看。

|    课程序号   |                              主题                              |                   可分租                            | 学习目标                                                                                                                  |                     课程链接                        |    作者                 |
| :-----------: | :------------------------------------------------------------: | :-------------------------------------------------: | ------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------: | :------------: |
|      01       |                机器学习介绍                                    |      [Introduction](1-Introduction/README.md)       | Learn the basic concepts behind machine learning                                                                                |   [lesson](1-Introduction/1-intro-to-ML/README.md)    |    Muhammad    |
|      02       |                机器学习历史                                    |      [Introduction](1-Introduction/README.md)       | Learn the history underlying this field                                                                                         |  [lesson](1-Introduction/2-history-of-ML/README.md)   |  Jen and Amy   |
|      03       |                机器学习和公平                                  |      [Introduction](1-Introduction/README.md)       | What are the important philosophical issues around fairness that students should consider when building and applying ML models? |     [lesson](1-Introduction/3-fairness/README.md)     |     Tomomi     |
|      04       |                机器学习用到的技术                              |      [Introduction](1-Introduction/README.md)       | What techniques do ML researchers use to build ML models?                                                                       | [lesson](1-Introduction/4-techniques-of-ML/README.md) | Chris and Jen  |
|      05       |                   回归介绍                                     |        [Regression](2-Regression/README.md)         | Get started with Python and Scikit-learn for regression models                                                                  |       [lesson](2-Regression/1-Tools/README.md)        |      Jen       |
|      06       |                北美南瓜价格 🎃                                 |        [Regression](2-Regression/README.md)         | Visualize and clean data in preparation for ML                                                                                  |        [lesson](2-Regression/2-Data/README.md)        |      Jen       |
|      07       |                北美南瓜价格 🎃                                 |        [Regression](2-Regression/README.md)         | Build linear and polynomial regression models                                                                                   |       [lesson](2-Regression/3-Linear/README.md)       |      Jen       |
|      08       |                北美南瓜价格 🎃                                 |        [Regression](2-Regression/README.md)         | Build a logistic regression model                                                                                               |      [lesson](2-Regression/4-Logistic/README.md)      |      Jen       |
|      09       |                          web应用 🔌                            |           [Web App](3-Web-App/README.md)            | Build a web app to use your trained model                                                                                       |        [lesson](3-Web-App/1-Web-App/README.md)        |      Jen       |
|      10       |                 分类介绍                                        |    [Classification](4-Classification/README.md)     | Clean, prep, and visualize your data; introduction to classification                                                            |  [lesson](4-Classification/1-Introduction/README.md)  | Jen and Cassie |
|      11       |             美味的亚洲和印度菜 🍜                              |    [Classification](4-Classification/README.md)     | Introduction to classifiers                                                                                                     | [lesson](4-Classification/2-Classifiers-1/README.md)  | Jen and Cassie |
|      12       |             美味的亚洲和印度菜 🍜                              |    [Classification](4-Classification/README.md)     | More classifiers                                                                                                                | [lesson](4-Classification/3-Classifiers-2/README.md)  | Jen and Cassie |
|      13       |             美味的亚洲和印度菜 🍜                              |    [Classification](4-Classification/README.md)     | Build a recommender web app using your model                                                                                    |    [lesson](4-Classification/4-Applied/README.md)     |      Jen       |
|      14       |                   聚类介绍                                     |        [Clustering](5-Clustering/README.md)         | Clean, prep, and visualize your data; Introduction to clustering                                                                |     [lesson](5-Clustering/1-Visualize/README.md)      |      Jen       |
|      15       |              探索尼日利亚音乐品味 🎧                           |        [Clustering](5-Clustering/README.md)         | Explore the K-Means clustering method                                                                                           |      [lesson](5-Clustering/2-K-Means/README.md)       |      Jen       |
|      16       |        自然语言处理介绍 ☕️                                     |   [Natural language processing](6-NLP/README.md)    | Learn the basics about NLP by building a simple bot                                                                             |    [lesson](6-NLP/1-Introduction-to-NLP/README.md)    |    Stephen     |
|      17       |                      常用NLP任务 ☕️                      |   [Natural language processing](6-NLP/README.md)    | Deepen your NLP knowledge by understanding common tasks required when dealing with language structures                          |           [lesson](6-NLP/2-Tasks/README.md)           |    Stephen     |
|      18       |             翻译和情感分析 ♥️              |   [Natural language processing](6-NLP/README.md)    | Translation and sentiment analysis with Jane Austen                                                                             |   [lesson](6-NLP/3-Translation-Sentiment/README.md)   |    Stephen     |
|      19       |                 欧洲浪漫酒店 ♥️                  |   [Natural language processing](6-NLP/README.md)    | Sentiment analysis with hotel reviews 1                                                                                         |      [lesson](6-NLP/4-Hotel-Reviews-1/README.md)      |    Stephen     |
|      20       |                  欧洲浪漫酒店 ♥️                  |   [Natural language processing](6-NLP/README.md)    | Sentiment analysis with hotel reviews 2                                                                                         |      [lesson](6-NLP/5-Hotel-Reviews-2/README.md)      |    Stephen     |
|      21       |            时间序列预测导论             |        [Time series](7-TimeSeries/README.md)        | Introduction to time series forecasting                                                                                         |    [lesson](7-TimeSeries/1-Introduction/README.md)    |   Francesca    |
|      22       | ⚡️ 世界能源使用 ⚡️ - ARIMA时间序列预测 |        [Time series](7-TimeSeries/README.md)        | Time series forecasting with ARIMA                                                                                              |       [lesson](7-TimeSeries/2-ARIMA/README.md)        |   Francesca    |
|      23       |             强化学习介绍             | [Reinforcement learning](8-Reinforcement/README.md) | Introduction to reinforcement learning with Q-Learning                                                                          |    [lesson](8-Reinforcement/1-QLearning/README.md)    |     Dmitry     |
|      24       |                 帮助彼得躲避狼 🐺                  | [Reinforcement learning](8-Reinforcement/README.md) | Reinforcement learning Gym                                                                                                      |       [lesson](8-Reinforcement/2-Gym/README.md)       |     Dmitry     |
|  Postscript   |            真实人工智能（ML）场景以及应用            |      [ML in the Wild](9-Real-World/README.md)       | Interesting and revealing real-world applications of classical ML                                                               |    [lesson](9-Real-World/1-Applications/README.md)    |      Team      |

## 离线学习

通过[Docsify](https://docsify.js.org/#/)可以离线学习本课程。首先Fork 这个repo，安装[Docsify] (https://docsify.js.org/#/quickstart)工具到自己的电脑上，在repo的根目录里执行"docsify server"命令， 通过"localhost:3000"就可以访问你本地的环境。

## PDFs

可以通过此[连接](pdf/readme.pdf)下载本教程的PDF版本.

## 参与贡献!

如果你想参与贡献翻译，请阅读 [翻译指南](TRANSLATIONS.md) 并在[这里](https://github.com/microsoft/ML-For-Beginners/issues/71)填写.

## 其他课程

我们团队也提供了以下其他课程:

- [Web Dev for Beginners](https://aka.ms/webdev-beginners)
- [IoT for Beginners](https://aka.ms/iot-beginners)
