from imageai.Prediction.Custom import ModelTraining
import sys
from os.path import isdir
import time

start_time = time.time()
dataset = sys.argv[1]
no_of_experiments = int(sys.argv[2])
model_type = sys.argv[3]
data_directory = ("data/" + dataset).format()
no_of_objects =2
if not isdir(data_directory):
    raise ValueError('Invalid data path given')
if dataset=='fruit':
    no_of_objects =30


model_trainer = ModelTraining()
if model_type =='SqueezeNet':
    model_trainer.setModelTypeAsSqueezeNet()
elif model_type =='DenseNet':
    model_trainer.setModelTypeAsDenseNet()
elif model_type =='InceptionV3':
    model_trainer.setModelTypeAsInceptionV3()
else:
    model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory(data_directory)
model_trainer.trainModel(num_objects=no_of_objects, num_experiments=no_of_experiments, enhance_data=True,batch_size=16,show_network_summary=True)
endtime = time.time()


print("The training time is: ", endtime-start_time)