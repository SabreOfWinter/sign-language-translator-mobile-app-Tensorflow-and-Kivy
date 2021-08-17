import os
import tensorflow as tf
import cv2
import numpy as np

def extract_videos():
    videos_train_path = '' #Directory for videos to be classified. Every video should be divided into folders for each class (E.g. './Data/Sign_hello/' would have all the videos of the sign hello) 
    frames_train_path = '' #Directory for extracted sequences of frames to be placed

    #Check for if frame folders have been created and create them if not
    for label in os.listdir(videos_train_path):
        if label not in os.listdir(frames_train_path):
            os.mkdir(frames_train_path + '/' + label)
    
    #Go through all training videos label by label to extract all frames 
    for label in os.listdir(videos_train_path):
        videos_path = videos_train_path + '/' + label
        frames_path = frames_train_path + '/' + label

        #Loop through each video in the current class to extract the frames 
        for video in os.listdir(videos_path):
            video_path = videos_path + '/' + video
            video_name = video.split('.')[0]

##########
    #Builds an array of the sequence of images
##########
def build_dataset(train_x, train_y, data_path):
    training_data = []
    number_of_classes = 0 #Number of video types
    number_of_samples = 0 #Number of videos

    for path in os.listdir(data_path): #Loop through each class
        number_of_classes += 1
        
        for video in os.listdir(data_path + '/' + path): #Each video
            current_frame = 0 #Track the number of frames processed in a video
            number_of_samples += 1 #Need to know how many samples there are to reshape the dataset array to fit the network
            train_y.append(number_of_classes - 1) #Minus 1 to maintain index range from 0 - x, while also counting the total number of classes

            for frame in os.listdir(data_path + '/' + path + '/' + video): #Each frame
                current_frame += 1
                img = cv2.imread(data_path + '/' + path + '/' + video + '/' + frame, cv2.IMREAD_GRAYSCALE)
                saved_img = np.array(img)
                saved_img = saved_img.astype(np.float32) / 255.0 #Normalize pixels of images
                training_data.append(saved_img) #Add frame in to training dataset

            train_x = np.array(training_data, dtype=object)

    return(train_x, train_y, number_of_samples, number_of_classes)

##########
    #Used to convert LRCN SavedModel format to .tflite format
##########
def convert_model():
    saved_model_dir = '' #path to SavedModel format of selected model to convert
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    # Save the model
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)

def main():
    pass

if __name__ == '__main__':
    main()