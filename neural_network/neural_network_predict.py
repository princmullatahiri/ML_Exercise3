from imageai.Prediction.Custom import CustomImagePrediction
import os
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import sys
import itertools



def save_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()




dataset = sys.argv[1]


if dataset=='fruit':
    path_arr = ['acerolas', 'apples', 'apricots', 'avocados', 'bananas', 'blackberries', 'blueberries', 'cantaloupes',
                'cherries', 'coconuts', 'figs', 'grapefruits', 'grapes',
                'guava', 'kiwifruit', 'lemons', 'limes', 'mangos', 'olives', 'oranges', 'passionfruit', 'peaches',
                'pears', 'pineapples', 'plums', 'pomegranates', 'raspberries',
                'strawberries', 'tomatoes', 'watermelons']
    num_objects =30
else:
    path_arr = ['car', 'no-car']
    num_objects=2

execution_path = os.getcwd()
#Need to create a way to get the variables from the sys.args method.
prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path, "data/" + dataset+ "/models/model.h5"))
prediction.setJsonPath(os.path.join(execution_path, "data/" + dataset + "/json/model_class.json"))
prediction.loadModel(num_objects=num_objects)

class_acc = [['class', 'accuracy']]
y_true = []
y_predict = []

all_predictions = [['prediction', 'percentage']]

for each_path in path_arr:

    print('============================' +each_path + " predictions==========================================")
    image_path = 'data/' +dataset+'/test/' + each_path
    all_images_array = []
    correct_predictions = 0
    incorrect_predictions = 0

    all_files = os.listdir(os.path.join(execution_path, image_path))
    for each_file in all_files:
        if(each_file.endswith(".jpg") or each_file.endswith(".png")):
            all_images_array.append(image_path + '/' + each_file)

    temp_y_true = []
    temp_y_predict = []

    results_array = prediction.predictMultipleImages(all_images_array, result_count_per_image=5)

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
micro_precision = metrics.precision_score(y_true, y_predict, average='micro')
micro_recall = metrics.recall_score(y_true, y_predict, average='micro')
micro_f1_score = metrics.f1_score(y_true, y_predict, average='micro')


macro_precision = metrics.precision_score(y_true, y_predict, average='macro')
macro_recall = metrics.recall_score(y_true, y_predict, average='macro')
macro_f1_score = metrics.f1_score(y_true, y_predict, average='macro')

cnf_matrix = metrics.confusion_matrix(y_true, y_predict)
#np.set_printoptions(precision=2)

            # Plot non-normalized confusion matrix
plt.figure()
save_confusion_matrix(cnf_matrix, classes=path_arr, title='Confusion matrix, without normalization')
plt.savefig("/Users/macbook/Documents/TU Wien/Sommer Semester 2019/Machine Learning/Exercise 3_new/ML_Exercise3/neural_network/reports/figures/"+dataset+"/confusion_matrix.png")

micro_results = [['accuracy', 'precision', 'recall', 'f1_score'], [accuracy, micro_precision, micro_recall, micro_f1_score]]
macro_results = [['precision', 'recall', 'f1_score'], [macro_precision, macro_recall, macro_f1_score]]

class_acc = np.array(class_acc)
class_acc = pd.DataFrame(data=class_acc[1:,0:], columns=class_acc[0,:])
class_acc.to_csv('./reports/performance_measures/'+dataset+'/class accuracy.csv', index=False)

micro_results = np.array(micro_results)
micro_results = pd.DataFrame(data=micro_results[1:, 0:], columns=micro_results[0, :])
micro_results.to_csv('./reports/performance_measures/'+dataset+'/micro_performance measures.csv', index=False)

macro_results = np.array(macro_results)
macro_results = pd.DataFrame(data=macro_results[1:, 0:], columns=macro_results[0, :])
macro_results.to_csv('./reports/performance_measures/'+dataset+'/macro_performance measures.csv', index=False)


all_predictions = np.array(all_predictions)
all_predictions = pd.DataFrame(data=all_predictions[1:,0:], columns=all_predictions[0,:])
all_predictions.to_csv('./reports/performance_measures/'+dataset+'/predictions.csv', index=False)


print("The confusion matrix as well as the results of this run have been stored under REPORTS directory ")




