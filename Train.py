import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import joblib


df = pd.read_csv('car_data.csv')
final_dataset=df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
final_dataset['Current Year'] = 2020
final_dataset['no_year']=final_dataset['Current Year']-final_dataset['Year']    
final_dataset.drop(['Year'],axis=1,inplace=True)
final_dataset.drop(['Current Year'],axis=1,inplace=True)
final_dataset=pd.get_dummies(final_dataset,drop_first=True)
X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0] 



model=ExtraTreesRegressor()
model.fit(X,y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


rf_random=RandomForestRegressor()


n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]

max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]  

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42,n_jobs=1)
rf_random.fit(X_train,y_train)
predictions=rf_random.predict(X_test)


# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')
            #Pickle file name

# dump information to that file
joblib.dump(rf_random, file)