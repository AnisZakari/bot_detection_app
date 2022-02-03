import pytest
from predict import make_basic_features, make_event_distribution_features, make_predictions
import pandas as pd
import numpy as np
import joblib
import os

path_to_model = os.path.join("model", "RandomForestModel.sav")
model =joblib.load(path_to_model)

#testing 
#--------------------------------------------------------------
@pytest.fixture
def input_data():
	input_columns = ["UserId", "Event", "Category"]
	input_data = [
				["03E7EE785DT", "click_carrousel", "Phone"],
				["F0F3098683T", "click_ad", "Leisure"],
				["5064A38F0DT", "click_carrousel", "Phone"],
				["5C8E90A354T", "click_carrousel", "Motor"],
				["DC1F29D286T", "send_sms", "Motor"],
				["2DA8AAA602T", "send_email", "Phone"],
				["281A8EE211T", "click_carrousel", "Phone"],
				["F866660C47T", "send_email", "Real_State"],
				["D2E47FF774T", "click_carrousel", "Phone"],
				["78C6D6B909T", "send_email", "Motor"]
			]
	return pd.DataFrame(input_data, columns = input_columns)

def test_make_basic_features(input_data):
	expected_cols = ['Event count', 'Event nunique', 'Category nunique']
	output = make_basic_features(input_data)
	assert np.isin(output.columns, expected_cols).all() == True
	assert len(output.columns) == 3

def test_make_event_distribution_features(input_data):
	expected_cols = ["click_ad", "click_carrousel", "phone_call", "send_email", "send_sms"]
	output = make_event_distribution_features(input_data)
	assert np.isin(output.columns, expected_cols).all() == True
	assert len(output.columns) > 1

#--------------------------------------------------------------
 
def test_make_predictions():
	input_pre_prediction_columns = ["Event_count", "Event_nunique", "Category_nunique", "click_ad", "click_carrousel", "phone_call", "send_email", "send_sms"]
	data_pre_predictions = [
							[19,  5,  2,  5,  3,  4,  4,  3],
						    [20,  5,  2,  5,  7,  4,  3,  1],
						    [21,  5,  4,  3,  4,  3,  4,  7],
						    [16,  5,  2,  3,  3,  3,  5,  2],
						    [21,  5,  5,  4,  7,  3,  3,  4]
						]
	datataframe_pre_predictions = pd.DataFrame(data_pre_predictions, columns = input_pre_prediction_columns)
	output = make_predictions(datataframe_pre_predictions, path_to_model)
	expected_cols = ["is_fake_probability", "UserId"]
	assert np.isin(expected_cols, output.columns).all() == True
	assert output.shape[1] == 2
	assert np.min(output["is_fake_probability"]) >= 0
	assert np.max(output["is_fake_probability"]) <= 1
	assert output.shape[0] == datataframe_pre_predictions.shape[0]



