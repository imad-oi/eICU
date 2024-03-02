# %%
import pandas as pd

# Charger les données
url = "https://physionet.org/files/eicu-crd-demo/2.0.1/"
patients_data = pd.read_csv(url + "patient.csv.gz")


# %%
patients_data

# %%
# Colonnes à supprimer (celles qui ne sont pas utiles)
colonnes_inutiles = ['patientunitstayid', 'patienthealthsystemstayid', 'hospitalid', 'wardid',
                     'hospitaladmittime24', 'hospitaladmitoffset', 'hospitaldischargeyear',
                     'hospitaldischargetime24', 'hospitaldischargeoffset', 'hospitaldischargelocation',
                     'unittype', 'unitadmittime24', 'unitadmitsource', 'unitvisitnumber', 'unitstaytype',
                     'dischargeweight', 'unitdischargetime24', 'unitdischargeoffset', 'unitdischargelocation',
                     'unitdischargestatus', 'uniquepid']

# Supprimer les colonnes inutiles
patients_data.drop(columns=colonnes_inutiles, inplace=True)

# %%
patients_data.info()

# %%
patients_data.head()

# %%
# drop rows with missing values
patients_data.dropna(inplace=True)


# %%
# Creating Dummy Variables

dummies = []
cols = ['gender','ethnicity','hospitaladmitsource']
for col in cols:
    dummies.append(pd.get_dummies(patients_data[col]))
    
dummies_df = pd.concat(dummies, axis=1)
dummies_df.head()

# %%
# extraire le Y et les x 

df = pd.concat([patients_data, dummies_df], axis=1)
df.drop(columns=cols, inplace=True)


# %%
df.head()

# %%
df.info()

# %%
# fill the missing values
df['admissionweight'] = df['admissionweight'].interpolate() # fill the missing values with the mean

df['age'] = df['age'].interpolate() # fill the missing values with the mean

df.info()

# %%
x = df.values 
y = df['hospitaldischargestatus'].values

# %%
import numpy as np

x = np.delete(x, 0, axis=1) # delete the target variable from the features

# %%
# Devide the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# %%



