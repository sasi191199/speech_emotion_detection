This is a speech emotion detection  model creation and web application  folder. It will train MLP model which consists of the RAVDESS dataset, and we uses that model to predict the output of the emotion  using the flask app. 


!!requirements:

To download the RAVDESS speech emotion recognition dataset:
================
 https://drive.google.com/file/d/1wWsrN2Ep7x6lWqOXfr4rpKGYrJhWc8z7/view
=====================

to install all our required dependencies open the terminal and run the below command

===========
. ./install_deps.sh
====================

it will create the virtual environment and populate all nessasary dependencies required

### MODEL:

A poly layer neural network  model to detect the emotion of the audio that we given in the .wav files.
to devolop or create the model see create_model.py
if  the create_model.py is fitted  to your app requirementts(emotions_to_observe, and path to sound data), simply run:

============
python3 create_model.py
=====================

to create the model.model binary file and test accuracy of your model


 APP:

Once the model.model binary is created, you can spin up the flask application (emotion_check):
To do so run

====================
. ./start_flask.sh
===============================

The app will run default on localhost:5000, the emotions available for predictions will correspond with the emotions_to_observe variable you have created  inside create_models.py (and are therefore available inside the model binary file)

