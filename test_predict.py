import pytest
from predict import make_basic_features, make_event_distribution_features, make_predictions
import pandas as pd
import numpy as np
import joblib
import os

path_to_model = os.path.join("model", "RandomForestModel.sav")
model =joblib.load(path_to_model)

# Testing first block of features --> basic features.
#--------------------------------------------------------------
@pytest.fixture
def input_data():
	input_data = [
				["03E7EE785DT", "click_carrousel", "Phone"],
				["F0F3098683T", "click_ad", "Leisure"],
				["5064A38F0DT", "click_carrousel", "Phone"],
				["5C8E90A354T", "click_carrousel", "Motor"],
				["5C8E90A354T", "send_sms", "Motor"],
				["DC1F29D286T", "send_sms", "Motor"],
				["2DA8AAA602T", "send_email", "Phone"],
				["03E7EE785DT", "click_carrousel", "Phone"],
				["F866660C47T", "send_email", "Real_State"],
				["D2E47FF774T", "click_carrousel", "Phone"],
				["03E7EE785DT", "send_email", "Motor"]
			]
	return pd.DataFrame(input_data, columns = ["UserId", "Event", "Category"])

def test_make_basic_features(input_data):

	data_expected = [
	                    ['03E7EE785DT', 3, 2, 2],
	                    ['2DA8AAA602T', 1, 1, 1],
	                    ['5064A38F0DT', 1, 1, 1],
	                    ['5C8E90A354T', 2, 2, 1],
	                    ['D2E47FF774T', 1, 1, 1],
	                    ['DC1F29D286T', 1, 1, 1],
	                    ['F0F3098683T', 1, 1, 1],
	                    ['F866660C47T', 1, 1, 1]
                ]
	expected_df = pd.DataFrame(data_expected, columns = ["UserId", "Event count", "Event nunique", "Category nunique"]).set_index("UserId")
	output_df = make_basic_features(input_data)
	assert output_df.equals(expected_df)



# Testing second block of features --> Event Distributions.
#--------------------------------------------------------------
def test_make_event_distribution_features(input_data):

	data_expected = [
						['03E7EE785DT', 0, 2, 1, 0],
				       	['2DA8AAA602T', 0, 0, 1, 0],
				       	['5064A38F0DT', 0, 1, 0, 0],
				       	['5C8E90A354T', 0, 1, 0, 1],
				    	['D2E47FF774T', 0, 1, 0, 0],
				    	['DC1F29D286T', 0, 0, 0, 1],
				    	['F0F3098683T', 1, 0, 0, 0],
				    	['F866660C47T', 0, 0, 1, 0]
       				]

	expected_df = pd.DataFrame(data_expected, columns = ["UserId", 'click_ad', 'click_carrousel', 'send_email', 'send_sms']).set_index("UserId")
	output_df = make_event_distribution_features(input_data)
	assert output_df.equals(expected_df)

# Testing Predictions
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
