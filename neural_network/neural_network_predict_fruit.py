from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path, "data/fruit/models/model_ex-074_acc-0.778409.h5"))
prediction.setJsonPath(os.path.join(execution_path, "data/fruit/json/model_class.json"))
prediction.loadModel(num_objects=30)


image_path = 'data/fruit/test/apples'
all_images_array = []

all_files = os.listdir(os.path.join(execution_path, image_path))
for each_file in all_files:
    if(each_file.endswith(".jpg") or each_file.endswith(".png")):
        all_images_array.append(image_path + '/' + each_file)


results_array = prediction.predictMultipleImages(all_images_array, result_count_per_image=3)

for each_result in results_array:
    predictions, percentage_probabilities = each_result["predictions"], each_result["percentage_probabilities"]
    for index in range(len(predictions)):
        print(predictions[index] , " : " , percentage_probabilities[index])
    print("-----------------------")