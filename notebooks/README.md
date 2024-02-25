```
данные лиц 500 людей CelebA-500. Они уже выровнены. Скачать можно тут: https://disk.yandex.ru/d/S8f03spLIA1wrw
Внутри:
- celebA_imgs — папка с выровненными картинками;
- celebA_anno.txt — файл с аннотацией — каждой картинке из celebA_imgs поставлен в соответствие ее id;
- celebA_train_split.txt — файл со сплитом на train/val/test.
Эти данные — часть открытого датасета CelebA, который один из стандартных для обучения моделей, связанных с лицами.
```


* train model on CE loss for classification task
* train model on ArcFace loss for classification task
* IR metric for both models and comparison similarity between faces on embeddings from models
* train model with face landmarks for face alignment
* faces detection and crop
* pipeline for faces recognition: detection -> alignment -> embeddings
