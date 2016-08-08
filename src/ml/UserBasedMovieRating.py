import MovieRating
import Constants
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

# User based movie ratings
# Used for recommend top movies for users
class UserBasedMovieRating(MovieRating):
	
	def __init__(self):
		self.name = "User Based Trainer"
		
		# create sparck context
		conf = SparkConf().setMaster(Constants.SPARK_MASTER).setAppName(Constants.APP_NAME)
		self.sc = SparkContext(conf = conf)
	
	# train user based model
	# save the model into resource folder
	def train(self):

		# load sample data into Spark RDD
		data = self.sc.textFile(Constants.RESOURCE_FOLDER + Constants.DATA_FILE)
		
		# process raw training data. 
		ratings = data.map(lambda x: x.split("::"))
					  .map(lambda x : Rating(int(x[0]), int(x[1]), float(x[2])))

		# build the recommendation model using ALS
		rank = 10 # 20, 30, 40, 50
		iterations = 10
		alpha = 0.01
		model = ALS.train(ratings, rank, iterations, alpha)

		#save model
		model.save(self.sc, Constants.RESOURCE_FOLDER + Constants.MODEL_FILE)
		
		
	# recommend top K movies for a given user
	def recommend(self, user, topK):
		
		# load model
		model = MatrixFactorizationModel.load(self.sc, Constants.RESOURCE_FOLDER + Constants.MODEL_FILE)
		model.recommendProducts(user, topK)
		
		#TODO: decide a model to return data
		