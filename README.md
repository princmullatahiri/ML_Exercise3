# Machine Learning Exercise 3

In this repository, you will find the implementation that the authors did as part of the Machine
Learning course. More specifically, the implementation for Topic 3.8 "Deep Learning"

## Getting Started

In order to be able to run this project, you must initially clone/download it and then install
all the prerequisites as required. 

### Prerequisites

Please make sure that you have Python 3 and pip3 installed in your local machine. Once you
have done that, please open a cmd(terminal) in the "ML_Exercise3" folder of the project
run the following command.

```
sudo pip3 install -r requirements.txt
```
This command will install all the packages that in the ```requirements.txt``` file.   
NOTE: The packages used have dependencies of their own which are not neccesarily included in 
the ```requirements.txt``` file. To check the dependencies of each package please look at 
```dependency_tree.txt``` file.

## Running the tests

The tests in this project are seperated into two main categories: 
* Traditional Methods 
* Neural Networks.
### Traditional Methods

The traditional method Scikit-learn was used. And the classifiers choosen are:
kNN, Naive Bayes, Decision Tree, Random Forest, SVC, LinearSVC. For using these classifiers feature extractor 
is very important because we use the csv created by ```featureExtractor.py```
For running the traditional approach use terminal:
```
python traditional_approach.py
```

### Neural Networks

In the neural networks category of tests you can either use the existing models to 
predict two types of images or learn a new model.

When predicting an image, please open a terminal(cmd) in the ```neural network``` folder
inside the project.

```
To be added
```

When trying to create a new model please run the following command in terminal.

```
python3 neural_network_train.py $dataset $num_experiments $model_type
```

* dataset - [car, fruit]
* num_experiments - Integer
* model_type - [SqueezeNet, DenseNet, InceptionV3, ResNet]

The models created by this will be stored in ```data/$dataset/models/ ```


For this implementation of deep neural networks we have used ImageAI package. In order to add
new instances to the existing data sets there is a structure that must be followed. For
more information please visit this [link](https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Prediction/CUSTOMTRAINING.md).


## Authors

* **Arlind Avdullahi** - [thedionysus](https://github.com/thedionysus)
* **Princ Mullatahiri** - [princmullatahiri](https://github.com/princmullatahiri)

## License

This project is for academic purposes only.


