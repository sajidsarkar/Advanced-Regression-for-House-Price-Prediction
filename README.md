# Advanced Regression for House Price Prediction
The project comprised of data exploration, feature engineering, and regression technique (Random Forest Classifier) to predict house sale prices. <ins>**Root Mean Square Error was reduced down to 0.18063.**</ins> Below is my description of the [source_code.ipynb:](https://github.com/sajidsarkar/Kaggle-House_Price-Advanced_Regression/blob/master/source_code.ipynb)


#### Importing necessary libraries

```ruby
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
pd.set_option("display.max_rows",None)
```
## Exploratory Data Analysis and Feature Engineering
#### Categorizing Different Features</H4>
Features have been categorized into the following for convenience:<br/>
 - **categorical_features** - features with data type as Object<br/>
 - **categroical_features_nan** - categorical features with presence of invalid values<br/>
 - **temporal_features** - features that contained date or time, such as year.<br/>
 - **numerical_features** - features with numeric values<br/>

```ruby
categorical_features = [feature for feature in train_data.columns if train_data[feature].dtypes == 'O']
categorical_features_nan = [feature for feature in train_data.columns if train_data[feature].dtypes == 'O'and train_data[feature].isnull().sum()>0]
temporal_features = [feature for feature in train_data.columns if 'Yr' in feature or 'Year' in feature]
numerical_features = [feature for feature in train_data.columns if train_data[feature].dtypes != 'O'and feature not in ['Id','SalePrice'] and feature not in temporal_features]
```

#### Heatmap to illustrate missing values in the overall dataset.
```ruby
plt.figure(figsize=(14,25))
plt.title('HeatMap illustrating distribution of NaN values for ALL features')
sns.heatmap(train_data.isnull().transpose(), cmap='YlGnBu', cbar_kws={'label': 'Missing'})
plt.show()
```
![sdf](https://user-images.githubusercontent.com/67841104/160978084-c9d732ab-4abe-498f-8b3c-6d4c75e3f6f4.PNG)



#### Impuration of Missing Values in Temporal Features
Only 'GarageYrBlt' feature has presence of invalid entries; therefore, relation of median SalePrice
with respect to both invalid and invalid entries in the feature was illustrated. Median SalePrice
with respect to invalid entries is lower than that of valid entries.
```ruby
data = train_data.copy()
data[data['GarageYrBlt'].notnull()].groupby('GarageYrBlt')['SalePrice'].median().plot()
plt.title('Median SalePrice with non-missing GarageYrBlt values')
plt.show()
data['GarageYrBlt'] = np.where(data['GarageYrBlt'].isnull(),'missing','not missing')
data.groupby('GarageYrBlt')['SalePrice'].median().plot.bar()
plt.show()
df_missing = data[data['GarageYrBlt']=='missing'][['GarageYrBlt','SalePrice']]
df_missing.boxplot(column='SalePrice')
plt.title('Distribution of median SalePirce with missing values')
plt.show()
df_notmissing = data[data['GarageYrBlt']=='not missing'][['GarageYrBlt','SalePrice']]
df_notmissing.boxplot(column='SalePrice')
plt.title('Distribution of median SalePirce with non-missing values')
plt.show()
```
![1](https://user-images.githubusercontent.com/67841104/160974463-65b731aa-1eae-4d64-9020-7e048145f9e8.png)
![2](https://user-images.githubusercontent.com/67841104/160974467-a149274a-9987-4ce1-83d0-c4b37dc2a771.png)</br>
![3](https://user-images.githubusercontent.com/67841104/160974476-adb040cd-44df-4417-98ff-a26f7de2b32a.png)
![4](https://user-images.githubusercontent.com/67841104/160974481-b923845b-9ac2-4060-a6e7-553e05f15fad.png)

From above visualizations, mdeian SalePrice with missing values is around USD 100,000, which is lower
than the median SalePrice, at around USD 170,000, with non-missing values. An assumption can be made on the GarageYrBlt value for median SalePrice of USD 100,000 by observing the line chart that shows median SalePrice with respect to non-missing GarageYrBlt values. In that plot, the range of years 1920 to 1940 has closest correspondence with median SalePrice of USD 100,000. Therefore, with this assumption, all missing values can be replaced by the value 1930.

```ruby
train_data['GarageYrBlt'] = np.where(train_data['GarageYrBlt'].isnull(),1930,train_data['GarageYrBlt'])
```

#### Transformation of Temporal Features
```ruby
for feature in temporal_features:
    data = train_data.copy()
    data.groupby(feature)['SalePrice'].median().plot()
    plt.show()
```

![1](https://user-images.githubusercontent.com/67841104/160974951-d27afa35-426f-4f21-bb2f-ba221191b391.png)
![2](https://user-images.githubusercontent.com/67841104/160974956-a7bed205-cea2-40d6-a7f3-334bca421c09.png)</br>
![3](https://user-images.githubusercontent.com/67841104/160974959-94ff252e-8ad0-4e55-87b8-af9f7a8e25b9.png)
![4](https://user-images.githubusercontent.com/67841104/160974964-c9d2264c-6438-41b4-99c8-c946f548e50e.png)

Year of build, remodelling, and garage build have positive trend with respect to sales price. However,
there is a negative trend of year sold with respect to sales price, which is unusual because a house's 
price should increase every year in a normal economy. This is occuring because the feature 'YrSold'
does not account for other three temporal features, as sales price at the year sold will depend on
how close build, remodelling and garage build took place relative to year sold.<br/><br/>
Therefore, all three temporal features, except 'YrSold', will be transformed by taking difference between the features with respect to year sold. After transformation of temporal features, the 'YrSold' feature will be dropped.

**Note:** In GarageYrBlt plot, there is a sharp dip at 1930, when compared to the same plot created when
analyzing missing values in GarageYrBlt. This is due to the replacement of missing values in this feature with 1930.

```ruby
for feature in temporal_features:
    if feature == 'YrSold': pass
    else:
        train_data[feature] = train_data['YrSold'] - train_data[feature]
train_data = train_data.drop('YrSold', axis = 1)
temporal_features.remove('YrSold')
for feature in temporal_features:
    data = train_data.copy()
    data.groupby(feature)['SalePrice'].median().plot()
    plt.xlabel('YrSold from '+feature)
    plt.title('Years between '+feature+' and YrSold')
    plt.show()
```
![1](https://user-images.githubusercontent.com/67841104/160975284-e1fca3f6-d94d-4a36-bce8-2362f7fa25a0.png)
![2](https://user-images.githubusercontent.com/67841104/160975290-dbf36da8-2783-4570-a9cf-d7158ec08a64.png)</br>
![3](https://user-images.githubusercontent.com/67841104/160975294-936efdc6-eaf4-47e7-916c-12692543b9ea.png)
The above line charts indicate that recently built or remodeled houses have higher sales price.

#### Imputation of Missing Values in Numerical Features
Missing data in continuous numerical features were replaced by the mean value of the feature.
Assumption was made that a missing value in continuous numerical feature could be represented
by the mean value of its feature's non-missing values.
```ruby
for feature in numerical_features:
    train_data[feature] = np.where(train_data[feature].isnull(), train_data[feature].mean(),
                                   train_data[feature])
```

The heatmap illustrates remaining missing values after imputation of missing values in numerical features.
```ruby
plt.figure(figsize=(14,25))
plt.title('HeatMap illustrating distribution of NaN values for ALL features')
sns.heatmap(train_data.isnull().transpose(), cmap='YlGnBu', cbar_kws={'label': 'Missing'})
plt.show()
```
![adswfdsf65454](https://user-images.githubusercontent.com/67841104/160977737-dcd05c8d-2852-4dc7-bebb-a0b543c9f8b8.PNG)


#### Imputation of Missing Data in Categorical Features
Visualizing NaN values in categorical features with a heatmap and quantifying number of missing values in each categorical feature.
```ruby
plt.figure(figsize=(10,6))
plt.title('HeatMap illustrating distribution of NaN values for categorical features')
sns.heatmap(train_data[categorical_features_nan].isnull().transpose(), cmap='YlGnBu', cbar_kws={'label': 'Missing'})
plt.show()

for feature in categorical_features_nan:
    data = train_data.copy()
    print('Respective % NaN values and # of unique values in '+feature+' are '+'{}'.format(np.round(data[feature].isnull().sum()/len(data),3))+' and '+'{}'.format(len(data[feature].unique())))
```
![adswfdsf65454](https://user-images.githubusercontent.com/67841104/160977824-d4c6c933-0280-46ca-887c-bc0a5a0efa8e.PNG)

Respective % NaN values and # of unique values in Alley are 0.938 and 3
Respective % NaN values and # of unique values in MasVnrType are 0.005 and 5
Respective % NaN values and # of unique values in BsmtQual are 0.025 and 5
Respective % NaN values and # of unique values in BsmtCond are 0.025 and 5
Respective % NaN values and # of unique values in BsmtExposure are 0.026 and 5
Respective % NaN values and # of unique values in BsmtFinType1 are 0.025 and 7
Respective % NaN values and # of unique values in BsmtFinType2 are 0.026 and 7
Respective % NaN values and # of unique values in Electrical are 0.001 and 6
Respective % NaN values and # of unique values in FireplaceQu are 0.473 and 6
Respective % NaN values and # of unique values in GarageType are 0.055 and 7
Respective % NaN values and # of unique values in GarageFinish are 0.055 and 4
Respective % NaN values and # of unique values in GarageQual are 0.055 and 6
Respective % NaN values and # of unique values in GarageCond are 0.055 and 6
Respective % NaN values and # of unique values in PoolQC are 0.995 and 4
Respective % NaN values and # of unique values in Fence are 0.808 and 5
Respective % NaN values and # of unique values in MiscFeature are 0.963 and 5

Categorical features containing more 40% of NaN values are dropped.
Rest of the categorical features with NaN values were recovered by replacing
them with the mode value.

```ruby
train_data.drop(['Alley','PoolQC','Fence','MiscFeature','FireplaceQu'],axis=1,inplace=True)
categorical_features.remove('Alley')
categorical_features.remove('PoolQC')
categorical_features.remove('Fence')
categorical_features.remove('MiscFeature')
categorical_features.remove('FireplaceQu')

for feature in categorical_features:
    train_data[feature] = np.where(train_data[feature].isnull(), train_data[feature].mode(), train_data[feature])
```


### Visualizing NaN values (if any) in overall training dataset using a heatmap.
```ruby
plt.figure(figsize=(14,25))
plt.title('HeatMap illustrating distribution of NaN values for ALL features')
sns.heatmap(train_data.isnull().transpose(), cmap='YlGnBu', cbar_kws={'label': 'Missing'})
plt.show()
```

![adswfdsf65454](https://user-images.githubusercontent.com/67841104/160977981-a7e6f821-5a5c-49c2-9660-c737c4efef08.PNG)


### Visualizing distribution of numerical features
```ruby
for feature in numerical_features:
    data = train_data.copy()
    data[feature].hist()
    plt.title(feature)
    plt.xlabel(feature)
    plt.show()
```

![1](https://user-images.githubusercontent.com/67841104/160975984-8e889451-d8a6-4f7f-92f3-bedbf18351da.png)
![2](https://user-images.githubusercontent.com/67841104/160976012-e604c27d-4f17-42e2-a621-57302e44fc31.png)
![3](https://user-images.githubusercontent.com/67841104/160976013-d8dd76e1-1d84-4623-a95a-683fc644cce1.png)
![4](https://user-images.githubusercontent.com/67841104/160976014-c0f9cf43-1f96-44f6-80cd-ee239a14887f.png)
![5](https://user-images.githubusercontent.com/67841104/160976016-1421c610-1141-45fd-bc4f-d522ead2cecf.png)
![6](https://user-images.githubusercontent.com/67841104/160976017-7d12bd44-db8f-414f-9fb6-0c911b5f2711.png)
![7](https://user-images.githubusercontent.com/67841104/160976019-c2cf914d-eba1-4521-99d9-7497e3246732.png)
![8](https://user-images.githubusercontent.com/67841104/160976021-0c75d9ee-015d-48f8-8299-4dcdc1486478.png)
![9](https://user-images.githubusercontent.com/67841104/160976022-729f3aad-2556-4034-8e72-3930fa617e75.png)
![10](https://user-images.githubusercontent.com/67841104/160976024-3231c93b-0ee3-47f8-95db-11619ecf1843.png)
![11](https://user-images.githubusercontent.com/67841104/160976025-af7de2a0-b717-4a90-be25-a276d0dae861.png)
![12](https://user-images.githubusercontent.com/67841104/160976027-546832c1-1979-45be-81ae-bfc12cabf3fd.png)
![13](https://user-images.githubusercontent.com/67841104/160976029-2f131e88-1734-4dc3-9a7a-cae67d6cf302.png)
![14](https://user-images.githubusercontent.com/67841104/160976030-252fd864-a526-4aac-93ef-df21acb38a03.png)
![15](https://user-images.githubusercontent.com/67841104/160976031-d0042469-31bc-4d6e-b725-356dff509b6a.png)
![16](https://user-images.githubusercontent.com/67841104/160976032-22064366-0fe9-4f76-9708-469ca78726ce.png)
![17](https://user-images.githubusercontent.com/67841104/160976033-f2f2f810-6a68-48e9-a21b-f97fe10a1060.png)
![18](https://user-images.githubusercontent.com/67841104/160976035-1d546491-dfa2-486d-a3dd-524328369600.png)
![19](https://user-images.githubusercontent.com/67841104/160976036-fc0423f1-a13c-4397-bb82-6e2cbf4e1fa6.png)
![20](https://user-images.githubusercontent.com/67841104/160976038-e45fe5c0-3e45-4bd8-86c0-609fcb904f37.png)
![21](https://user-images.githubusercontent.com/67841104/160976039-86f38750-0c21-4fc7-b22e-ed10d6ecb658.png)
![22](https://user-images.githubusercontent.com/67841104/160976040-0650640e-1aa2-44c8-b462-2a011eb7c47a.png)
![23](https://user-images.githubusercontent.com/67841104/160976041-32ce0032-e0ae-473c-ad20-785451ab7bf3.png)
![24](https://user-images.githubusercontent.com/67841104/160976042-22aa839f-5ed5-48d4-95ea-4d6a93cdc87b.png)
![25](https://user-images.githubusercontent.com/67841104/160976043-e8948f29-bef3-4866-89a2-f13110a731be.png)
![26](https://user-images.githubusercontent.com/67841104/160976045-f09d3136-966d-4720-88fa-7b3a278d083e.png)
![27](https://user-images.githubusercontent.com/67841104/160976047-dba83b44-fdf8-4d6a-aed3-5e07a8dca002.png)
![28](https://user-images.githubusercontent.com/67841104/160976048-2d35a5a8-cb0e-43a0-8107-7fea8fb4ce1a.png)
![29](https://user-images.githubusercontent.com/67841104/160976049-16a235d6-b47a-4ad6-87ea-8a2bbe9484b4.png)
![30](https://user-images.githubusercontent.com/67841104/160976050-4d1737fc-aae3-4710-87bb-195665b2ce53.png)
![31](https://user-images.githubusercontent.com/67841104/160976052-4041d8f5-5acd-4cc4-bf4c-28f10d9fb0ef.png)
![32](https://user-images.githubusercontent.com/67841104/160976053-d7a30558-ca9c-4096-980b-f3c3e5ffccb8.png)


#### Log Transformation on Skewed Features
As can be seen from above distributions, 'MSSubClass','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','WoodDeckSF','OpenPorchSF', and 'EnclosedPorch' features are positively skewed; therefore, natural log transformation was applied to transoform their distributions close to normal distribution.
```ruby
numerical_features_skewed = ['MSSubClass','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','WoodDeckSF','OpenPorchSF','EnclosedPorch']
for feature in numerical_features_skewed:
        train_data[feature] = np.where(train_data[feature] == 0, train_data[feature], np.log(train_data[feature]))
```

#### Performing standardization on numerical features
Although Random Forest Classifier will be used, features should still go through appropriate transoformation and scaling since it is a regression problem.
```ruby
for feature in numerical_features:
    train_data[feature] = (train_data[feature] - train_data[feature].mean())/train_data[feature].std()
```
#### Performing similar data cleaning and transformations on testing data
```ruby
test_data = pd.read_csv('test.csv')
categorical_features = [feature for feature in test_data.columns if test_data[feature].dtypes == 'O']
categorical_features_nan = [feature for feature in test_data.columns if test_data[feature].dtypes == 'O'and test_data[feature].isnull().sum()>0]
temporal_features = [feature for feature in test_data.columns if 'Yr' in feature or 'Year' in feature]
numerical_features = [feature for feature in test_data.columns if test_data[feature].dtypes != 'O'and feature not in ['Id','SalePrice'] and feature not in temporal_features]
test_data['GarageYrBlt'] = np.where(test_data['GarageYrBlt'].isnull(),1930,test_data['GarageYrBlt'])
for feature in temporal_features:
    if feature == 'YrSold': pass
    else:
        test_data[feature] = test_data['YrSold'] - test_data[feature]
        
test_data = test_data.drop('YrSold', axis = 1)
temporal_features.remove('YrSold')
for feature in numerical_features:
    test_data[feature] = np.where(test_data[feature].isnull(), test_data[feature].mean(),
                                   test_data[feature])
test_data.drop(['Alley','PoolQC','Fence','MiscFeature','FireplaceQu'],axis=1,inplace=True)
categorical_features.remove('Alley')
categorical_features.remove('PoolQC')
categorical_features.remove('Fence')
categorical_features.remove('MiscFeature')
categorical_features.remove('FireplaceQu')

for feature in categorical_features:
    test_data[feature] = np.where(test_data[feature].isnull(), test_data[feature].mode(), test_data[feature])

numerical_features_skewed = ['MSSubClass','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','WoodDeckSF','OpenPorchSF','EnclosedPorch']
for feature in numerical_features_skewed:
        test_data[feature] = np.where(test_data[feature] == 0, test_data[feature], np.log(test_data[feature]))
for feature in numerical_features:
    test_data[feature] = (test_data[feature] - test_data[feature].mean())/test_data[feature].std()
test_id = test_data['Id']
x_test = test_data.drop('Id',axis=1)
```
## Training and Prediction Using Random Forest Classifier
#### Preparing training and test data and assinging dummy values to categorical features
Both x test and x train datasets were combined for creating dummy variables. This was to ensure both test and train datasets contained exact features after applying "pd.get_dummies" function. This can occur when a categorical feature in train set has different categories inside it when compared to the same feature in test set.
```ruby
x_train = train_data.drop(['Id','SalePrice'],axis=1)
y_train = train_data['SalePrice']
x_train['train'] = 1
x_test['train'] = 0
combined = pd.concat([x_train, x_test])
dummies = pd.get_dummies(combined[categorical_features], prefix='dummy')
combined.drop(categorical_features,axis=1,inplace=True)
combined = pd.concat([combined,dummies],axis=1)
x_train = combined[combined['train']==1]
x_test = combined[combined['train']==0]
```
#### Using RandomForestClassifier to predict SalePrice
```ruby
model = RandomForestClassifier(n_estimators=2000)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
y_predict = pd.DataFrame(y_predict, columns=['SalePrice'])
y_predict.insert(0,'Id',test_id)
y_predict.to_csv('submission.csv',index=False)
```
