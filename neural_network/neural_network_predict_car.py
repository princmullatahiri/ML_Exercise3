from imageai.Prediction.Custom import CustomImagePrediction
import os
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from pathlib import Path
from traditional_approach import save_confusion_matrix

execution_path = os.getcwd()
#Need to create a way to get the variables from the sys.args method.
prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path, "data/car/models/model.h5"))
prediction.setJsonPath(os.path.join(execution_path, "data/car/json/model_class.json"))
prediction.loadModel(num_objects=2)


path_arr = ['car', 'no-car']

class_acc = [['class', 'accuracy']]
y_true = []
y_predict = []

all_predictions = [['prediction', 'percentage']]

for each_path in path_arr:

    print('============================' +each_path + " predictions==========================================")
    image_path = 'data/car/test/' + each_path
    all_images_array = []
    correct_predictions = 0
    incorrect_predictions = 0

    all_files = os.listdir(os.path.join(execution_path, image_path))
    for each_file in all_files:
        if(each_file.endswith(".jpg") or each_file.endswith(".png")):
            all_images_array.append(image_path + '/' + each_file)

    temp_y_true = []
    temp_y_predict = []

    results_array = prediction.predictMultipleImages(all_images_array, result_count_per_image=2)

    for each_result in results_array:
        predictions, percentage_probabilities = each_result["predictions"], each_result["percentage_probabilities"]
        temp_y_true.append(each_path)
        temp_y_predict.append(predictions[0])
        y_true.append(each_path)
        y_predict.append(predictions[0])
        for index in range(len(predictions)):
            print(predictions[index], " : ", percentage_probabilities[index])
            all_predictions.append([predictions[index], percentage_probabilities[index]])
        print("-----------------------")
        all_predictions.append(['====', '===='])
    accuracy = metrics.accuracy_score(temp_y_true, temp_y_predict)
    class_acc.append([each_path, accuracy])
    print('=====================================================================================================')
accuracy = metrics.accuracy_score(y_true, y_predict)
precision = metrics.precision_score(y_true, y_predict, average='micro')
recall = metrics.recall_score(y_true, y_predict, average='micro')
f1_score = metrics.f1_score(y_true, y_predict, average='micro')

cnf_matrix = metrics.confusion_matrix(y_true, y_predict)
#np.set_printoptions(precision=2)

            # Plot non-normalized confusion matrix
plt.figure()
save_confusion_matrix(cnf_matrix, classes=path_arr, title='Confusion matrix, without normalization')
plt.savefig("/Users/macbook/Documents/TU Wien/Sommer Semester 2019/Machine Learning/Exercise 3_new/ML_Exercise3/neural_network/reports/figures/car/confusion_matrix.png")

results = [['accuracy', 'precision', 'recall', 'f1_score'], [accuracy, precision, recall, f1_score]]

class_acc = np.array(class_acc)
class_acc = pd.DataFrame(data=class_acc[1:,0:], columns=class_acc[0,:])
class_acc.to_csv('./reports/performance_measures/car/class accuracy.csv', index=False)

results = np.array(results)
results = pd.DataFrame(data=results[1:,0:], columns=results[0,:])
results.to_csv('./reports/performance_measures/car/performance measures.csv', index=False)


all_predictions = np.array(all_predictions)
all_predictions = pd.DataFrame(data=all_predictions[1:,0:], columns=all_predictions[0,:])
all_predictions.to_csv('./reports/performance_measures/car/predictions.csv', index=False)