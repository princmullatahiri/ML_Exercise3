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
#absolutePath = "C:/Princi/TU Wien/Semestri 1/Machine Learning/Exercises/Exercise 3 - New/"
absolutePath = "/Users/macbook/Documents/TU Wien/Sommer Semester 2019/Machine Learning/Exercise 3_new/"
def classify_with_gridsearch(dataset):
    """
        Implemented with GridSearchCV()
        :return:
        """

    if dataset == 'fruit':
        # Target names
        target_names = ['acerolas', 'apples', 'apricots', 'avocados', 'bananas', 'blackberries', 'blueberries',
                        'cantaloupes', 'cherries', 'coconuts', 'figs', 'grapefruits', 'grapes', 'guava',
                        'kiwifruit', 'lemons', 'limes', 'mangos', 'olives', 'oranges', 'passionfruit', 'peaches',
                        'pears', 'pineapples', 'plums', 'pomegranates', 'raspberries', 'strawberries', 'tomatoes',
                        'watermelons']
        # Target labels
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                  28, 29]
        path = Path(absolutePath + 'ML_Exercise3/data/Interim/fruits')
    else:
        # Target names
        target_names = ['negative', 'positive']
        # Target labels
        labels = [0, 1]
        path = Path(absolutePath + 'ML_Exercise3/data/Interim/cars')
    # load data sets
    print(target_names[1])
    print('Loading CSVs ({}).'.format(str(datetime.datetime.now())))
    # TODO: Look into why the relative paths aren't working
    #path = Path(absolutePath + 'ML_Exercise3/data/Interim/fruits')
    data = pd.read_csv(path / 'data.csv', index_col=0)
    dataOpenCV_1D = pd.read_csv(path / 'dataOpenCV_1D.csv', index_col=0)
    dataOpenCV_2D = pd.read_csv(path / 'dataOpenCV_2D.csv', index_col=0)
    dataOpenCV_3D = pd.read_csv(path / 'dataOpenCV_3D.csv', index_col=0)

    print('... done with loading CSVs ({}).'.format(str(datetime.datetime.now())))

    # And now we finally classify
    # these are our feature sets; we will use each of them individually to train classifiers
    trainingSets = {
        "PillowData": data,
        "dataOpenCV_1D": dataOpenCV_1D,
        "dataOpenCV_2D": dataOpenCV_2D,
        "dataOpenCV_3D": dataOpenCV_3D,
    }

    # set up a number of classifiers
    classifiers = {
        "KNN": neighbors.KNeighborsClassifier(),
        "Naive Bayes": naive_bayes.GaussianNB(),
        "Decision Tree": tree.DecisionTreeClassifier(),
        "Random Forest": ensemble.RandomForestClassifier(),
        "SVC": svm.SVC(),
        "LinearSVC": svm.LinearSVC(),
    }

    # set up parameter grids for each classifier
    parameters = {
        "KNN": {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        "Naive Bayes": {},
        "Decision Tree": {'min_samples_split': range(10, 500, 50), 'max_depth': range(1, 20, 5)},
        "Random Forest": {'n_estimators': range(200, 2000, 200)},
        "SVC": {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1]},
        "LinearSVC": {'C': [0.001, 0.01, 0.1, 1, 10]},
    }

    # Now iterate over the datasets & classifiers, and train...
    print('Training Models ({}).'.format(str(datetime.datetime.now())))
    for index, df in trainingSets.items():
        for name, classifier in classifiers.items():
            # Splitting the data into train and test data using stratify
            # More info on stratify here:
            # https://stackoverflow.com/questions/35472712/how-to-split-data-on-balanced-training-set-and-test-set-on-sklearn
            y = df['class']
            X = df.drop('class', axis=1)

            # Train Test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=69, stratify=y)

            # Hyperparameter tune
            #For me it doesnt work with -1 change n_jobs to 1
            clf = GridSearchCV(classifier, cv=5, param_grid=parameters[name], n_jobs=-1, verbose=1,
                               return_train_score=False)

            # Fit
            clf.fit(X, y)

            # Print classification report
            y_pred = clf.best_estimator_.predict(X_test)
            report = classification_report(y_pred, y_test, output_dict=True)

            # Create df from report
            p = Path(absolutePath + 'ML_Exercise3/reports/performance_measures/' + dataset + '/')
            dataframe = pd.DataFrame.from_dict(report)
            dataframe.to_csv(p / '{}_{}.csv'.format(index, name))

            p = Path(absolutePath + 'ML_Exercise3/reports/cv_results/' + dataset + '/')
            dataframe = pd.DataFrame.from_dict(clf.cv_results_)
            dataframe.to_csv(p / '{}_{}.csv'.format(index, name))

            # Open and append best params and best score
            with open(dataset + "_scores_gridsearch.dat", 'a') as f:
                f.write('{}_{}, {}, {}\n'.format(index, name, clf.best_params_, clf.best_score_))

            # Compute confusion matrix
            cnf_matrix = confusion_matrix(y_test, y_pred)
            np.set_printoptions(precision=2)

            # Plot non-normalized confusion matrix
            plt.figure()
            save_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion matrix, without normalization')
            p = Path(absolutePath + 'ML_Exercise3/reports/figures/' + dataset + '/')
            plt.savefig(p / '{}_{}_Confusion_Matrix.png'.format(index, name))



    print('... done with training Models ({}).'.format(str(datetime.datetime.now())))

def save_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

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
    classify_with_gridsearch('car')



if __name__ == '__main__':
    main()
    sys.exit(0)
