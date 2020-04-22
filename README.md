# Disaster Response Pipeline Project

### Purpose
Create a webapp that uses a multioutput machine learning model, built on flask, and uploaded to Github. 
The code should demonstrate ability to use Github, show ability to conduct ETL, building a Machine learning pipeline using NLP techniques, and the barebones of deploying a webapp locally. 


### Requirements:
The python version used is 3.6.3. The rest of the dependencies are included in requirements.txt. I formed the requirements to be minimal change form the packages that we were given in the class IDE. 

Data is not included since this should be run on proprietary data. 


### Sources:
All content in this project came from the documentation of the libraries used, the classes in udacity, and previous experience. 


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
