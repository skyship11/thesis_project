# Analysis of bias and fairness in facial emotion recognition

Our study focuses on reducing the bias and improving the fairness of various CNN models in Facial Emotion Recognition (FER). 

We choose the RAF-DB dataset to study and train. **'rafdb_aligned.csv'** includes different labels for each image, including true emotion, gender, race, and age group.

Emotion: 1: Surprise, 2: Fear, 3: Disgust, 4: Happiness, 5: Sadness, 6: Anger, 7: Neutral;

Gender: 0: male, 1: female; 

Race: 0: Caucasian, 1: African-American, 2: Asian; 

Age (5 ranges): 0: 0-3, 1: 4-19,	2: 20-39,	3: 40-69,	4: 70+.

### The standard CNN model and VGG model
We use a simple CNN architecture and a VGG architecture to train and get 30 models respectively. we call these two models, the standard CNN model and VGG model.
In **'algorithms'** folder, these two trained model files are in **'baseline'** and **'VGG'** folders each.
The two **'.npy' files under the name 'Adam'** contain confusion matrices and test accuracies for 30 standard CNN models, and **two '.npy' files under the name 'VGG'** include those of 30 VGG models.

### SMOTE application
The application of SMOTE is useful for improving model fairness. 
The **'algorithms/VGG+SMOTE_model.pt'** is saved as the model file. And **'algorithms/Smote...npy'** contains the confusion matrix of VGG+SMOTE model.

### TL application
For Transfer Learning (TL) method, we have three frameworks and use two external dataset (FER2013 and SFEW). 
**'fer2013_modified.csv'** is the samples of FER2013 dataset, and we use the emotion label number of RAF-DB dataset to replace original number.
The SFEW dataset is available from 'Kaggle'.

For each dataset used in TL, we have three separate TL models, with the model files and performance performance files in the **'algorithms/TL_FER'** folder and **'algorithms/TL_SFEW'** folder, respectively.

### Code implementation
**'CNN_model.py'** includes the code of all model implementation. **'statistic_plot_total.py'** is the statistic analysis of the RAF-DB dataset.
