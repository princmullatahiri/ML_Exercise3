import sys
import pandas as pd
import datetime
import os
from pathlib import Path
from sklearn import neighbors, naive_bayes, tree, ensemble, svm
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, cross_validate, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, make_scorer, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools


#Arlind change absoulute Path for you
absolutePath = "C:/Princi/TU Wien/Semestri 1/Machine Learning/Exercises/Exercise 3 - New/"
#absolutePath = "/Users/macbook/Documents/TU Wien/Sommer Semester 2019/Machine Learning/Exercise 3_new/"

# Target names FOR FUITS
# target_names = ['acerolas', 'apples', 'apricots', 'avocados', 'bananas', 'blackberries', 'blueberries',
#                 'cantaloupes', 'cherries', 'coconuts', 'figs', 'grapefruits', 'grapes', 'guava',
#                 'kiwifruit', 'lemons', 'limes', 'mangos', 'olives', 'oranges', 'passionfruit', 'peaches',
#                 'pears', 'pineapples', 'plums', 'pomegranates', 'raspberries', 'strawberries', 'tomatoes',
#                 'watermelons']
# # Target labels
# labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
#           28, 29]
# Target names FOR CARS
target_names = ['negative', 'positive']
# Target labels
labels = [0, 1]

def accuracy_score(y_true, y_pred):


    report = classification_report(y_true, y_pred, target_names=target_names, labels=labels, output_dict=True)
    print(report)

    with open('store.dat', 'r') as f:
        lines = f.read().splitlines()
        last_line = lines[-1]

    # Uncomment if for fruits and comment cars
    #p = Path(absolutePath + 'ML_Exercise3/reports/performance_measures/fruit/without_grid_search/')
    p = Path(absolutePath + 'ML_Exercise3/reports/performance_measures/car/without_grid_search/')
    df = pd.DataFrame.from_dict(report)

    if os.path.isfile(p / '{}'.format(last_line)):
        old_df = pd.read_csv(p / '{}'.format(last_line), index_col=0)
        result = pd.concat([old_df, df])
        result.to_csv(p / '{}'.format(last_line))
    else:
        df.to_csv(p / '{}'.format(last_line))

    return accuracy_score(y_true, y_pred)


def classification_with_no_gridsearch(dataset):

    print('Load data ({}).'.format(str(datetime.datetime.now())))
    path = Path(absolutePath + 'ML_Exercise3/data/Interim/' + dataset + 's/')
    data = pd.read_csv(path / 'data.csv', index_col=0)
    dataOpenCV_1D = pd.read_csv(path / 'dataOpenCV_1D.csv', index_col=0)
    dataOpenCV_2D = pd.read_csv(path / 'dataOpenCV_2D.csv', index_col=0)
    dataOpenCV_3D = pd.read_csv(path / 'dataOpenCV_3D.csv', index_col=0)

    print('done loading data ({}).'.format(str(datetime.datetime.now())))

    trainingSets = {
        "PillowData": data,
        "dataOpenCV_1D": dataOpenCV_1D,
        "dataOpenCV_2D": dataOpenCV_2D,
        "dataOpenCV_3D": dataOpenCV_3D,
    }

    classifiers = {
        "KNN": neighbors.KNeighborsClassifier(),
        "Naive Bayes": naive_bayes.GaussianNB(),
        "Decision Tree": tree.DecisionTreeClassifier(),
        "Random Forest": ensemble.RandomForestClassifier(),
        "SVC": svm.SVC(),
        "LinearSVC": svm.LinearSVC(),
    }

    for index, df in trainingSets.items():
        for name, classifier in classifiers.items():
            with open("store.dat", 'w') as f:
                f.write('{}_{}.csv\n'.format(index, name))


            sss = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=69)
            y = df['class']
            X = df.drop('class', axis=1)
            nested_score = cross_val_score(classifier, X, y, cv=sss,
                                           scoring=make_scorer(accuracy_score))

            with open(dataset + "_scores.dat", 'a') as f:
                f.write('{}_{}: {}\n'.format(index, name, nested_score))

def classification_with_gridsearch(dataset):


    if dataset == 'fruit':
        target_names = ['acerolas', 'apples', 'apricots', 'avocados', 'bananas', 'blackberries', 'blueberries',
                        'cantaloupes', 'cherries', 'coconuts', 'figs', 'grapefruits', 'grapes', 'guava',
                        'kiwifruit', 'lemons', 'limes', 'mangos', 'olives', 'oranges', 'passionfruit', 'peaches',
                        'pears', 'pineapples', 'plums', 'pomegranates', 'raspberries', 'strawberries', 'tomatoes',
                        'watermelons']
        path = Path(absolutePath + 'ML_Exercise3/data/Interim/fruits')
    else:
        target_names = ['negative', 'positive']
        path = Path(absolutePath + 'ML_Exercise3/data/Interim/cars')
    print(target_names[1])
    print('Loading data ({}).'.format(str(datetime.datetime.now())))
    # TODO: Look into why the relative paths aren't working
    #path = Path(absolutePath + 'ML_Exercise3/data/Interim/fruits')
    data = pd.read_csv(path / 'data.csv', index_col=0)
    dataOpenCV_1D = pd.read_csv(path / 'dataOpenCV_1D.csv', index_col=0)
    dataOpenCV_2D = pd.read_csv(path / 'dataOpenCV_2D.csv', index_col=0)
    dataOpenCV_3D = pd.read_csv(path / 'dataOpenCV_3D.csv', index_col=0)

    print('done loading data ({}).'.format(str(datetime.datetime.now())))

    trainingSets = {
        "PillowData": data,
        "dataOpenCV_1D": dataOpenCV_1D,
        "dataOpenCV_2D": dataOpenCV_2D,
        "dataOpenCV_3D": dataOpenCV_3D,
    }

    classifiers = {
        "KNN": neighbors.KNeighborsClassifier(),
        "Naive Bayes": naive_bayes.GaussianNB(),
        "Decision Tree": tree.DecisionTreeClassifier(),
        "Random Forest": ensemble.RandomForestClassifier(),
        "SVC": svm.SVC(),
        "LinearSVC": svm.LinearSVC(),
    }

    parameters = {
        "KNN": {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        "Naive Bayes": {},
        "Decision Tree": {'min_samples_split': range(10, 500, 50), 'max_depth': range(1, 20, 5)},
        "Random Forest": {'n_estimators': range(200, 2000, 200)},
        "SVC": {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1]},
        "LinearSVC": {'C': [0.001, 0.01, 0.1, 1, 10]},
    }

    print('Running classifiers ({}).'.format(str(datetime.datetime.now())))
    for index, df in trainingSets.items():
        for name, classifier in classifiers.items():
            y = df['class']
            X = df.drop('class', axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=69, stratify=y)
            #For me it doesnt work with -1 change n_jobs to 1
            clf = GridSearchCV(classifier, cv=5, param_grid=parameters[name], n_jobs=-1, verbose=1, return_train_score=False)
            clf.fit(X, y)
            y_pred = clf.best_estimator_.predict(X_test)
            report = classification_report(y_pred, y_test, output_dict=True)

            p = Path(absolutePath + 'ML_Exercise3/reports/performance_measures/' + dataset + '/')
            dataframe = pd.DataFrame.from_dict(report)
            dataframe.to_csv(p / '{}_{}.csv'.format(index, name))

            p = Path(absolutePath + 'ML_Exercise3/reports/cv_results/' + dataset + '/')
            dataframe = pd.DataFrame.from_dict(clf.cv_results_)
            dataframe.to_csv(p / '{}_{}.csv'.format(index, name))

            with open(dataset + "_scores_gridsearch.dat", 'a') as f:
                f.write('{}_{}, {}, {}\n'.format(index, name, clf.best_params_, clf.best_score_))

            cnf_matrix = confusion_matrix(y_test, y_pred)
            np.set_printoptions(precision=2)

            plt.figure()
            save_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion matrix, without normalization')
            p = Path(absolutePath + 'ML_Exercise3/reports/figures/' + dataset + '/')
            plt.savefig(p / '{}_{}_Confusion_Matrix.png'.format(index, name))



    print('done with classifying ({}).'.format(str(datetime.datetime.now())))

def save_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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


def main():
    #classify_with_gridsearch('fruit')
    #classify_with_gridsearch('car')
    #classification_with_no_gridsearch('fruit')
    classification_with_no_gridsearch('car')


if __name__ == '__main__':
    main()
    sys.exit(0)
