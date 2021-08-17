import os
import tensorflow as tf

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