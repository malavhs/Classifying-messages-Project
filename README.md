# Classifying-messages-Project

This project classifies the messages input by the user into multiple categories (36). 

The Model being used in this project as part of the pipeline is the AdaBoost Classifier which is part of the ensemble ML algorithms with the estimator as the DecisionTreeClassifier.

# How to run:

1. To clean data, go inside the /data folder (cd data/)
  -- run (python process_data.py disaster_messages.csv disaster_categories.csv DisasterR.db)

2. Next, to train the model, go to the models folder (cd ../models/)
  -- run (python python train_classifier.py ../data/DisasterR.db classifier.pkl)
  
3. Finally, run the web app, go to app directory (cd ../app/)
  -- run (python run.py) 

Now, the web app will be Running on http://0.0.0.0:3000/ on your local machine.

# File descriptions:

1. process_data.py (This file cleans up the data -- removes duplicates, coverts values to binary etc. and stores it inside a SQLite database)

2. train_classifier.py (This file tokenizes the data, trains the model, evaluates the model as well as exports it into a pickle file)

3. run.py (It has the code for the visualisations as well as it runs the web app on the localhost)

4. ML Pipeline Preparation.ipynb - it contains various explorations and also contains a GridSearchCV code, along with other things.

5. DisasterR.db - It is the SQLite database.

6. classifier.pkl - It is the pickle file of the trained model.

