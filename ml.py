import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import pyodbc
import csv

def is_csv_empty(file_path):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            return False
    return True

if (is_csv_empty('./train.csv') and  is_csv_empty('./test.csv')):
    #LocalDb
    server = 'TestServer'
    database = 'TrainData'

    # Query
    sql_query = 'Select * from dbo.train'
    sql_query_test = 'Select * from dbo.test'

    #Connect
    connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes'
    connection = pyodbc.connect(connection_string)
    cursor = connection.cursor()

    # Execute 
    cursor.execute(sql_query)

    #fetch
    rows = cursor.fetchall()

    # Close  connection
    connection.close()


    # Write data to CSV file
    csv_file_path = './train.csv'
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header
        csv_writer.writerow([column[0] for column in cursor.description])
        
        # Write data
        csv_writer.writerows(rows)

    print(f'Data has been exported to {csv_file_path}')

    #Connection #2
    connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes'
    connection = pyodbc.connect(connection_string)
    cursor2 = connection.cursor()
    cursor2.execute(sql_query_test)
    rows_test = cursor2.fetchall()

    csv_file_path_test = './test.csv'
    with open(csv_file_path_test, 'w', newline='') as csvfile2:
        csv_writer = csv.writer(csvfile2)
        
        # Write header
        csv_writer.writerow([column[0] for column in cursor2.description])
        
        # Write data
        csv_writer.writerows(rows_test)

    print(f'Data has been exported to {csv_file_path_test}')

    connection.close()

df = pd.read_csv('./train.csv')
print(df.head())
df.shape
print(df.describe())
print(df.info())
plt.style.use('default')
plt.figure(figsize = (12,6))
sns.heatmap(df.corr())
plt.show()

# #gin/akkumlyator haraberutyun
plt.figure(figsize = (12,6))
sns.barplot(x = 'price_range', y = 'battery_power', data=df)
plt.show()

# #haraberutyun gin/laynutyun/barzrutyun
plt.figure(figsize = (14,6))
plt.subplot(1,2,1)
sns.barplot(x = 'price_range', y = 'px_height', data=df, palette = 'Reds', hue = 'price_range', legend = False)
plt.subplot(1,2,2)
sns.barplot(x = 'price_range', y = 'px_width', data=df , palette = 'Blues', hue = 'price_range', legend = False)
plt.show()

# #ram/gin
plt.figure(figsize = (12,6))
sns.barplot(x = 'price_range', y = 'ram',data=df)
plt.show()

# #gin/3g/4g
plt.figure(figsize = (12,6))
sns.countplot(data = df, x= 'three_g' , hue = df['price_range'] , palette = 'pink')
plt.show()

plt.figure(figsize = (12,6))
sns.countplot(data = df, x = 'four_g', hue = df['price_range'] , palette = 'Blues')
plt.show()

# #gin/memory
plt.figure(figsize = (12,6))
sns.lineplot(x = 'price_range' , y = 'int_memory' , data = df , hue = 'dual_sim')
plt.show()

# #datayi-preprocessing
x = df.drop(['price_range'] , axis = 1)
y = df['price_range']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 0)

# #k-nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(x_train,y_train)

from sklearn.decomposition import PCA
# Fit PCA to your training data
# Reduce to 2 dimensions for visualization
pca = PCA(n_components=2)  
x_train_pca = pca.fit_transform(x_train)

# # Plot the reduced data
plt.figure(figsize=(8, 6))
plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=y_train, cmap=plt.cm.Paired)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization of Training Data')
plt.show()

predictions = knn.predict(x_test)


test_df = pd.read_csv('./test.csv')
print(test_df.head())
print(test_df.shape)
test_df = test_df.drop(['id'] , axis = 1)
print(test_df.shape)
test_pred = knn.predict(test_df)
test_df['predicted_price'] = test_pred
print(test_df.head())
print(test_df['predicted_price'])
accuracy = accuracy_score(y_test , predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

import xgboost as xg  

xgbModel = xg.XGBClassifier()
xgbModel.fit(x_train,y_train)
xgbPredictions = xgbModel.predict(x_test)
accuracy_score(y_test,xgbPredictions)


XGBtest_df = pd.read_csv('./test.csv')
print(XGBtest_df.head())
print(XGBtest_df.shape)
XGBtest_df = XGBtest_df.drop(['id'] , axis = 1)
print(XGBtest_df.shape)
XGBtest_pred = xgbModel.predict(XGBtest_df)
XGBtest_df['predicted_price'] = XGBtest_pred
print(XGBtest_df.head())
print(XGBtest_df['predicted_price'])
XGBAccuracy = accuracy_score(y_test,xgbPredictions)
print("XGB Accuracy %.2f%%" % (XGBAccuracy * 100.0))

