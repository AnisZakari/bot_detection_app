import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
import argparse
import os
import joblib

def parse_args():
	parser = argparse.ArgumentParser(description='Model training script',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--inputcsv", help="path to the input csv", required=True)
	parser.add_argument("--modelpath", help="path to the model", default=os.path.join("model", "RandomForestModel.sav"))
	parser.add_argument("--outputcsv", help="path to the output csv")
	args = parser.parse_args()
	return args


def read_inputcsv(csv_path):
	return pd.read_csv(csv_path, index_col = 0)

# This function is used for aggregation. It returns a list of every clicks made by a user (ex: ["click_carrousel", "send_email", "send_email"...])
# It is used in the function make_event_distribution_features.
def get_list(series, column = "Event"):
    return " ".join(series)

# This function creates the first block of features
def make_basic_features(dataset):
	df = dataset.groupby("UserId")[["Event", "Category"]].agg({"Event":["count", "nunique"], "Category": "nunique"})
	df.columns = [" ".join(item) for item in df.columns] # getting rid of Multi-indexing caused by groupby
	return df


# This function creates the second block of features; for each user it gives the event distribution. 
def make_event_distribution_features(dataset):
	vectorizer = CountVectorizer()
	corpus = dataset.groupby("UserId")["Event"].agg(get_list)
	#The behaviour of groupby may change depending on python version
	try:
		X = vectorizer.fit_transform(corpus)
	except:
		X = vectorizer.fit_transform(corpus["Event"])
	event_distribution = pd.DataFrame(X.todense(), columns = vectorizer.get_feature_names_out(), index = corpus.index)
	return event_distribution

#Wrapping up the 2 previous functions (The 2 blocks of features)
def make_features(dataset):
	basic_features = make_basic_features(dataset)
	event_distribution_features = make_event_distribution_features(dataset)
	#Merging the 2 blocks of features.
	X = basic_features.merge(event_distribution_features, left_index = True, right_index = True)
	return X


def make_predictions(df, path_to_model):
	scaler = StandardScaler()
	X = scaler.fit_transform(df)
	model =joblib.load(path_to_model)
	y_probs = model.predict_proba(X)[:,1] # Probability of being Fake
	y_probs_df = pd.DataFrame({"UserId": df.index, "is_fake_probability": y_probs})
	return y_probs_df


def main(args):
	#getting filename from path
	filename = os.path.basename(args.inputcsv)
	dataset = read_inputcsv(args.inputcsv)
	#making the design matrix
	X = make_features(dataset)
	#Computing output and saving it.
	y_prob = make_predictions(X, args.modelpath)
	#if outputcsv arg is given we'll use it, otherwise it will be saved in the output folder with the same name as given in input
	if args.outputcsv is None:
		output_path = os.path.join("output", filename)
	else:
		output_path = args.outputcsv

	y_prob.to_csv(output_path, index = False)
	print(">>>>")
	print("> Predictions have successfully been made and saved to", output_path)
	print("> Recommended threshold to use is 0.5")

if __name__ == "__main__":
	args = parse_args()
	main(args)




