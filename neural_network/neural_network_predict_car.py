from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()
#Need to create a way to get the variables from the sys.args method.
prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path, "data/car/models/model_ex-016_acc-0.726562.h5"))
prediction.setJsonPath(os.path.join(execution_path, "data/car/json/model_class.json"))
prediction.loadModel(num_objects=2)


path_arr = ['car', 'no-car']



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


    results_array = prediction.predictMultipleImages(all_images_array, result_count_per_image=2)

    for each_result in results_array:
        predictions, percentage_probabilities = each_result["predictions"], each_result["percentage_probabilities"]
        if(predictions[0]==each_path):
            correct_predictions += 1
        else:
            incorrect_predictions += 1
        for index in range(len(predictions)):
            print(predictions[index] , " : " , percentage_probabilities[index])
        print("-----------------------")
    print("Number of correct predictions " + str(correct_predictions))
    print("Number of incorrect predictions " + str(incorrect_predictions))
    total = correct_predictions + incorrect_predictions
    print('Total number of tests ' + str(total))
    print('=====================================================================================================')