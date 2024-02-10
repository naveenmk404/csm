# pip install scikit-surprise
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the dataset
data = Dataset.load_builtin('ml-100k')

# Define a reader
reader = Reader(rating_scale=(1, 5))

# Load the dataset using the reader
data = Dataset.load_builtin('ml-100k')

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.25)

# Define the algorithm - User-based Collaborative Filtering with KNNBasic
algo = KNNBasic(k=40, sim_options={'user_based': True})

# Train the algorithm on the training set
algo.fit(trainset)

# Make predictions on the testing set
predictions = algo.test(testset)

# Evaluate the performance of the model using RMSE
rmse = accuracy.rmse(predictions)

