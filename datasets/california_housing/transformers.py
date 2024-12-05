from sklearn.base import BaseEstimator, TransformerMixin
from .distance_to_ocean import distance_to_coast

import numpy as np
import pandas as pd


class DistanceToCoast(BaseEstimator, TransformerMixin):
	"""
	Transformer which adds distance to coast using the longitude and latitude
	Expects X is a np array.
	Initialised with:
		- longitude_index: int
		- latitude_index: int
	the column indices of the longitude and latitude values.
	"""
	def __init__(self, longitude_index: int, latitude_index: int):
		super().__init__()
		self.longitude_index = longitude_index
		self.latitude_index = latitude_index

	def fit(self, X, y=None):
		return self
	
	def transform(self, X: np.ndarray):
		distances = np.array(
			[
				distance_to_coast(lon, lat) 
				for lon, lat in X[:, (self.longitude_index, self.latitude_index)]
			]
		).reshape(-1, 1)
		
		return np.concat([X, distances], axis=1)
	

class DistanceToCoastPD(BaseEstimator, TransformerMixin):
	"""
	Transformer which adds distance to coast using the longitude and latitude
	Expects X is a DataFrame.
	"""
	def fit(self, X, y=None):
		return self
	
	def transform(self, X: pd.DataFrame):
		distances = X.apply(
			lambda _: distance_to_coast(_["longitude"], _["latitude"]), 
			axis=1
		)
		
		return pd.merge(
			left=X,
			right=pd.DataFrame(distances, columns=["distance_to_coast"]),
			left_index=True,
			right_index=True
		)
	

class RoomsPerHouse(BaseEstimator, TransformerMixin):
	"""
	Transformer which adds custom attribute `rooms_per_house`
	"""
	def __init__(self, total_rooms_index: int, households_index: int):
		super().__init__()
		self.total_rooms_index = total_rooms_index
		self.households_index = households_index

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		_ = X[:, self.total_rooms_index] / X[:, self.households_index]
		return np.concat([
			X, _.reshape(-1, 1)
		], axis=1)
	

class RoomsPerHousePD(BaseEstimator, TransformerMixin):
	"""
	Transformer which adds custom attribute `rooms_per_house`
	"""
	def fit(self, X, y=None):
		return self

	def transform(self, X):
		a = X["total_rooms"] / X["households"]
		return pd.merge(
			X,
			pd.DataFrame(a, columns=["rooms_per_house"]),
			left_index=True,
			right_index=True
		)
	

class BedroomsPerHouse(BaseEstimator, TransformerMixin):
	"""
	Transformer which adds custom attribute `bedrooms_per_house`
	"""
	def __init__(self, total_bedrooms_index: int, households_index: int):
		super().__init__()
		self.total_bedrooms_index = total_bedrooms_index
		self.households_index = households_index

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		_ = X[:, self.total_bedrooms_index] / X[:, self.households_index]
		return np.concat([
			X, _.reshape(-1, 1)
		], axis=1)
	

class BedroomsPerHousePD(BaseEstimator, TransformerMixin):
	"""
	Transformer which adds custom attribute `bedrooms_per_house`
	"""
	def __init__(self):
		super().__init__()


	def fit(self, X, y=None):
		return self

	def transform(self, X):
		a = X["total_bedrooms"] / X["households"]
		return pd.merge(
			X,
			pd.DataFrame(a, columns=["bedrooms_per_house"]),
			left_index=True,
			right_index=True
		)
	

class PopulationPerHouse(BaseEstimator, TransformerMixin):
	"""
	Transformer which adds custom attribute `population_per_house`
	"""
	def __init__(self, population_index: int, households_index: int):
		super().__init__()
		self.population_index = population_index
		self.households_index = households_index

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		_ = X[:, self.population_index] / X[:, self.households_index]
		return np.concat([
			X, _.reshape(-1, 1)
		], axis=1)
	

class PopulationPerHousePD(BaseEstimator, TransformerMixin):
	"""
	Transformer which adds custom attribute `population_per_house`
	"""

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		a = X["population"] / X["households"]
		return pd.merge(
			X,
			pd.DataFrame(a, columns=["population_per_house"]),
			left_index=True,
			right_index=True
		)
	

class CorrectColumns(BaseEstimator, TransformerMixin):
	"""
	Transformer which strips the columns we do not want to include in the classification
	"""
	def __init__(self, keep_column_indices: tuple[int, ...]):
		super().__init__()
		self.keep_column_indices = keep_column_indices

	def fit(self, X, y=None):
		return self
	
	def transform(self, X):
		return X[:, self.keep_column_indices]
	

_keep_columns = (
	"distance_to_coast",
	"rooms_per_house",
	"bedrooms_per_house",
	"population_per_house"
)

class CorrectColumnsPD(BaseEstimator, TransformerMixin):
	"""
	Transformer which strips the columns we do not want to include in the classification
	"""
	def fit(self, X, y=None):
		return self
	
	def transform(self, X):
		return X[_keep_columns]
	