# ANN_Model_SolarPanel-Classification
Overview 
This project aims to detect faults in solar panels (e.g., dusty, dirty, bird droppings, etc.) through image 
classification using a Convolutional Neural Network (CNN). Once a fault is detected, a speech 
notification is played, and a reminder is set to notify the user to check the solar panels after a fixed 
time interval (e.g., 1 hour). The system also generates synthetic fault images for testing. 

#Link to Dataset: https://www.kaggle.com/code/pythonafroz/dust-detection-on-solar-panel-using-inceptionv3

Key Components: 
1. Image Classification Model: Trains a model to classify solar panel images into clean or dusty 
categories. 
2. Synthetic Image Generation: Generates synthetic images to simulate faulty solar panels. 
3. Notification and Reminder System: Notifies the user when a fault is detected, and sets a 
reminder after a specified time.

##1. Image Classification Model  
This code implements a Fully Connected Neural Network (ANN) for binary image classification, 
specifically to detect whether a solar panel is dusty or clean.

###Data Loading and Preprocessing: 
The function load_images_from_folder is responsible for loading and processing images from a folder 
structure. The dataset has subfolders representing the two classes "clean" and "dusty"—and the 
function reads all images in these folders. Each image is loaded using OpenCV’s cv2.imread(), resized 
to a standard size of 150x150 pixels using cv2.resize(), and normalized by scaling the pixel values to a 
range of [0, 1]. NumPy arrays. 

###Label Encoding and Data Splitting: 
The labels, which are strings ("clean" and "dusty"), are encoded into numeric values using 
LabelEncoder from Scikit-learn. The images and labels are then split into training and testing sets 
using train_test_split from Scikit-learn, with 80% of the data allocated for training and 20% for 
testing.

###Model Definition: 
A Sequential Neural Network model is defined using Keras. The model consists of several layers: 
1. Flatten: The input layer reshapes the image data (150x150x3) into a 1D vector. 
2. Dense Layers: These fully connected layers have ReLU activation functions, which introduce 
non-linearity into the model and allow it to learn more complex relationships. The first two 
Dense layers have 256 and 128 neurons respectively, followed by a 64-neuron layer. 
3. Dropout Layer: A dropout rate of 0.5 is applied after the first Dense layer to prevent 
overfitting by randomly deactivating half of the neurons during training. 
4. Output Layer: The output layer consists of a single neuron with a sigmoid activation function, 
which outputs a value between 0 and 1, representing the probability that the input image is 
dusty (1) or clean (0).

###Model Training and Evaluation: 
The model is trained using the training data (X_train, y_train) for 10 epochs with a batch size of 32. 
During training, the model's performance is also evaluated on the test data (X_test, y_test) to monitor 
its ability to generalize. The fit function trains the model and the evaluate function computes the final 
test accuracy after training. The accuracy of the model is printed as a percentage. 

###Saving the Model: 
After training and evaluation, the model is saved to a file called solar_panel_fault_model.keras using 
the model.save() function.

![image](https://github.com/user-attachments/assets/bf7fa723-0d23-4b13-8cdf-58cfccb5b727)

###Output: 
The printed output shows the test accuracy of the trained model, indicating how well the model 
performs in detecting dusty vs. clean solar panels on the test dataset, expressed as a percentage. 


##2. Synthetic Image Generation 
This code defines a function generate_synthetic_images that is used to generate a set of synthetic 
images with various faults, simulating issues commonly found in solar panels .  
Image Transformation and Fault Generation: 
1. Gaussian Blur: The function first applies a random Gaussian blur to the image with a 15x15 
kernel. 
2. Noise Addition: Next, the code adds random noise to the image. The noise is generated using 
a normal distribution with a mean of 0 and a standard deviation of 0.5, which introduces 
visual distortions resembling sensor noise or dirt on the panel's surface. 
3. Brightness Adjustment: The image brightness is then altered randomly.  
4. Snow Effect: The function simulates the presence of snow on the solar panel by randomly 
adding "snowflakes." 
5. Bird Droppings: The function also generates random bird droppings on the image by adding 
white circular patches to the image.

###Model Prediction and Visualization: 
After generating the synthetic images, the model is used to predict whether each image is clean or 
dusty.  The predicted label and confidence score are displayed as titles on the images, and the images are 
saved to the specified output folder. Each image is converted to an 8-bit format and saved as a JPEG 
file.

##3. Reminder and Notification System  
This code detects dust on solar panels and sends audio notifications to alert users when a fault is 
detected.

###Speech Notification: 
The play_speech_notification function uses the Google Text-to-Speech library to convert a provided 
text message into speech. This feature is used to deliver audio alerts, providing a hands-free way of 
notifying users of potential issues.

###Fault Detection and Notification: 
The core function, notify_if_fault_detected, is responsible for detecting faults in the solar panel 
images. It takes a prediction value, which represents the model's confidence in the presence of a fault. 
If the prediction exceeds a threshold of 0.5, it is assumed that the panel is "Dusty" and a fault is 
detected. A message is printed to the console, and the play_speech_notification function is called to 
deliver an audio warning to the user. Additionally, a reminder is set to trigger 1 hour later, alerting the 
user to check the solar panel again. 

###Loading and Preprocessing Images: 
The load_generated_images_from_folder function loads images from a specified folder and 
preprocesses them for input into the machine learning model. It resizes the images to 150x150 pixels 
and normalizes the pixel values to be between 0 and 1. The function also assigns labels to the images 
based on their filenames (assuming the folder structure distinguishes between "dusty" and "clean" 
images).

###Indefinite Loop: 
The while True loop at the end of the code ensures that the program continues running indefinitely, 
which is necessary to allow the timer for reminders to function properly. The loop includes a 
time.sleep(1) command, which prevents the program from using too much CPU by introducing a brief 
pause between iterations.
![image](https://github.com/user-attachments/assets/003811e7-5447-4a5d-af52-1237e542d4ab)

