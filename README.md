# music-genre-classification
Classifying the genre of music songs by their track features (e.g. popularity, loudness, danceability, time)

Among the methods used are KNN, LDA, SVM, GBDT etc.
Models were stacked using a forward-stepwise selection process.
Evaluation via F1-Score and ROC-AUC


Model | Accuracy | Precision | Recall | F1-Score | 
--- | --- | --- | --- |--- |
Logit | 0.8229 | 0.3765 | 0.4692 | 0.4178 | 
DT  | 0.8276 | 0.3918 | 0.4952 | 0.4375 | 
RF | 0.8570 | 0.4714| 0.4624 | 0.4669 | 
XGB  | 0.8548 | 0.4689 | 0.5458 | 0.5044 | 
Stack  | 0.8713 | 0.5251 | 0.5144 | 0.5197 | 
GB  | 0.8613 | 0.4896 | 0.5773 | 0.5298 | 


ROC Curve of selected models:

<img src="https://github.com/kaimaiworm/music-genre-classification/assets/70534743/8489c77d-8f38-47af-bdbd-3c22850f6f3b" width="500">






Data: 

https://www.kaggle.com/datasets/purumalgi/music-genre-classification
