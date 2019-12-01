# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path

#Code starts here 
data = pd.read_csv(path)
data['Gender'].replace('-', 'Agender',inplace=True)
gender_count = data['Gender'].value_counts()
gender_count.plot(kind="bar")



# --------------
#Code starts here
alignment = data['Alignment'].value_counts()
alignment.plot(kind="pie")
plt.title("Character Alignment")


# --------------
#Code starts here
sc_df = data[['Strength','Combat']]
cov_df = sc_df.cov()
sc_covariance = cov_df['Strength']['Combat']
sc_strength = sc_df['Strength'].std()
sc_combat = sc_df['Combat'].std()
sc_pearson = sc_covariance/(sc_combat*sc_strength)
print(sc_pearson)

ic_df = data[['Intelligence','Combat']]
cov_df = ic_df.cov()
ic_covariance = cov_df['Intelligence']['Combat']
ic_intelligence = ic_df['Intelligence'].std()
ic_combat = ic_df['Combat'].std()
ic_pearson = ic_covariance/(ic_combat*ic_intelligence)
print(ic_pearson)


# --------------
#Code starts here
total_highdf = data.quantile(0.99)
total_high = total_highdf['Total']
super_best = data[data['Total'] > total_high]
super_best_names = [super_best['Name']]
print(super_best_names)


# --------------
#Code starts here
ax_1 = super_best['Intelligence'].plot(kind="box")
plt.title("Intelligence")

ax_2 = super_best['Speed'].plot(kind="box",ax=ax_1)
plt.title("Speed")

ax_3 = super_best['Power'].plot(kind="box",ax=ax_2)
plt.title("Power")


