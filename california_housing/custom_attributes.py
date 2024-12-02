import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


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
	