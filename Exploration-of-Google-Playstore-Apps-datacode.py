# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here
data = pd.read_csv(path)
data['Rating'].plot(kind='hist')

data = data[data['Rating'] <= 5]
data['Rating'].plot(kind='hist')
#Code ends here


# --------------
# code starts here
total_null = data.isnull().sum()
percent_null = total_null/data.isnull().count()


missing_data = pd.concat([total_null,percent_null],keys=['Total','Percent'],axis=1)
# print(missing_data)

data.dropna(inplace=True)

total_null_1 = data.isnull().sum()
percent_null_1 = total_null_1/data.isnull().count()
missing_data_1 = pd.concat([total_null_1,percent_null_1],keys=['Total','Percent'],axis=1)
print(missing_data_1)
# code ends here


# --------------

#Code starts here
sns.catplot(x="Category",y="Rating",data=data, kind="box", height = 10)


#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
data['Installs'] = data['Installs'].str.replace('+', '')
data['Installs'] = data['Installs'].str.replace(',', '').astype(int)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Installs'] = le.fit_transform(data['Installs'])
# print(data['Installs'])
sns.regplot(x="Installs", y="Rating", data=data)
#Code ends here



# --------------
#Code starts here
# print(data['Price'].value_counts())
data['Price'] = data['Price'].str.replace('$', '').astype(float)
sns.regplot(x="Price", y="Rating", data=data)
#Code ends here


# --------------

#Code starts here
# print(data['Genres'].unique())
data['Genres'] = data['Genres'].apply(lambda x: (x.split(';'))[0])
gr_mean = data[['Genres','Rating']].groupby('Genres',as_index=False).mean()
# print(gr_mean)
gr_mean = gr_mean.sort_values('Rating')
print(gr_mean)
#Code ends here


# --------------

#Code starts here
data['Last Updated'] = pd.to_datetime(data['Last Updated'])
# print(data['Last Updated'])
max_date = data['Last Updated'].max()
data['Last Updated Days'] = max_date - data['Last Updated']
data['Last Updated Days'] = data['Last Updated Days'].dt.days
sns.regplot(x="Last Updated Days", y="Rating", data=data)
#Code ends here


