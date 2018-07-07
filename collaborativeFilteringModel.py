import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from ..recommendation.init import interactions_train_df
#from website.myproject.myapp.Models.recommendation.init import interactions_train_df
print("********Collaborative Filtering Model********")
#Creating a sparse pivot table with users in rows and items in columns

users_items_pivot_matrix_df = interactions_train_df.pivot(index='personId',
                                                          columns='contentId',
                                                          values='eventStrength').fillna(0)

users_items_pivot_matrix_df.head(10)

users_items_pivot_matrix = users_items_pivot_matrix_df.as_matrix()
users_items_pivot_matrix[:10]

users_ids = list(users_items_pivot_matrix_df.index)
users_ids[:10]

#The number of factors to factor the user-item matrix.
NUMBER_OF_FACTORS_MF = 15
#Performs matrix factorization of the original user item matrix
U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)

U.shape

Vt.shape

sigma = np.diag(sigma)
sigma.shape

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
all_user_predicted_ratings

#Converting the reconstructed matrix back to a Pandas dataframe
cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
cf_preds_df.head(10)

len(cf_preds_df.columns)

