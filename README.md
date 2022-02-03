# Bot_Detection

# Usage
Recommended python version: `python3.7`  
To run this project see the following steps:

# Creating virtual environment:  
`conda create --name bot_detection python=3.7`  
then:  
`conda activate bot_detection`

# Clone repo:  
`git clone https://github.com/AnisZakari/bot_detection_app.git`

# Accessing the folder:  
`cd bot_detection_app`  

# Install Requirements:
`pip install -r requirements.txt`


# Make predictions
You can make predictions quickly with the following command:  
`python3 predict.py --inputcsv data/fake_users_test.csv`

- The default behaviour is to use the model `RandomForestModel.sav` which is in the model folder.
- If no output path is specified the prediction file will be saved in the output folder with the same name as given in input

## Specify an output path
you can specify an output path with `--outputcsv`:    
`python3 predict.py --inputcsv data/fake_users_test.csv --outputcsv output/my_output.csv`

## Use a custom model
you can use a custom model with `--modelpath` you have to specify the model path:  
`python3 predict.py --inputcsv data/fake_users_test.csv --model "model/RandomForestModel.sav" --outputcsv output/my_output.csv`

Note that preferably the model has to come from  `scikit-learn==1.0.2`. It also has to be saved with the library joblib. A trained model can be saved like this:
```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(X, y)

path_to_model = "..."
with open(path_to_model, "wb") as f:
    joblib.dump(clf, f)
```

# Run Tests
```
pytest test_predict.py
```