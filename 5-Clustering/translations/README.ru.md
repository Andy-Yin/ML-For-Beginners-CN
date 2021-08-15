# Модели кластеризации для машинного обучения

Кластеризация - это задача машинного обучения, при которой она ищет объекты, которые похожи друг на друга, и группирует их в группы, называемые кластерами. Что отличает кластеризацию от других подходов в машинном обучении, так это то, что все происходит автоматически, и справедливо сказать, что это противоположность supervised learning.

## Региональная тема: модели кластеризации для музыкальных вкусов нигерийской публики 🎧

У разнообразной публики Нигерии самые разные музыкальные вкусы. Использование данных, извлеченных из Spotify (на основе [этой статьи](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421), давайте посмотрим на музыку, популярную в Нигерии. Этот набор данных включает данные о различных песнях "танцевальность", "акустичность", "громкость", "речевость", "популярность" и "энергия". Будет интересно обнаружить закономерности в этих данных!

![Поворотный стол](./images/turntable.jpg)

Фото <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText"> Марсела Ласкоски </a> на <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText"> Unsplash </a>
  
В этой серии уроков вы откроете для себя новые способы анализа данных с помощью методов кластеризации. Кластеризация особенно полезна, когда в наборе данных отсутствуют метки. Если на нем есть ярлыки, тогда могут быть более полезными методы классификации, подобные тем, которые вы изучили на предыдущих уроках. Но в случаях, когда вы хотите сгруппировать немаркированные данные, кластеризация - отличный способ обнаружить закономерности.

> Существуют полезные инструменты с небольшим количеством кода, которые могут помочь вам узнать о работе с моделями кластеризации. Попробуйте [Azure ML для этой задачи](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-15963-cxa)
## Уроки

1. [Введение в кластеризацию](1-Visualize/README.md)
2. [Кластеризация K-Means](2-K-Means/README.md)
## Благодарности

Эти уроки были написаны с помощью 🎶 [Джен Лупер](https://www.twitter.com/jenlooper) с полезными отзывами [Ришит Дагли](https://rishit_dagli) и [Мухаммад Сакиб Хан Инан](https://twitter.com/Sakibinan).

Набор данных [Нигерийские песни](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) был получен из Kaggle, как и из Spotify.

Полезные примеры K-Means, которые помогли в создании этого урока, включают [исследование радужной оболочки глаза](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), [вводный блокнот](https://www.kaggle.com/prashant111/k-means-clustering-with-python) и [пример гипотетической НПО](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).