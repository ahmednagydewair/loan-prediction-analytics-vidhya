import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
train = df.iloc[: , :-1]
train['label'] = '11'
target = df['Loan_Status']
test = pd.read_csv('test.csv')
test['label'] = '22'

###################combine two dataframes together
combined = train.append(test)
combined.reset_index(inplace=True)
combined.drop(['index', 'Loan_ID'], inplace=True, axis=1)

########### fill na for all values on the data set 
combined['Gender'] = combined['Gender'].fillna(value='Male')
combined['Married'] = combined['Married'].fillna(value='Yes')
combined['Dependents'] = combined['Dependents'].fillna(value='0')
combined['Self_Employed'] = combined['Self_Employed'].fillna(value='No')
combined['LoanAmount'] = combined['LoanAmount'].fillna(value=129)
combined['Loan_Amount_Term'] = combined['Loan_Amount_Term'].fillna(value=360)
combined['Credit_History'] = combined['Credit_History'].fillna(value=1)

################# transform all values t numeric 
from sklearn.preprocessing import LabelEncoder
l1 = LabelEncoder()
combined['Gender'] = l1.fit_transform(combined['Gender'])
l2 = LabelEncoder()
combined['Married'] = l2.fit_transform(combined['Married'])
l3 = LabelEncoder()
combined['Education'] = l3.fit_transform(combined['Education'])
l4 = LabelEncoder()
combined['Self_Employed'] = l4.fit_transform(combined['Self_Employed'])
l5 = LabelEncoder()
combined['Property_Area'] = l5.fit_transform(combined['Property_Area'])

l_label = LabelEncoder()
target = l5.fit_transform(target)

def change_to_int(x):
    if x == '3+':
        return 3
    else:
        return int(x)

combined['Dependents'] = combined['Dependents'].map(change_to_int )

##### standard scalar to all the data frame 

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
combined.iloc[:,:-1] = sc.fit_transform(combined.iloc[:,:-1])

from sklearn.model_selection import train_test_split
x = combined[combined['label']=='11'].iloc[:,:-1]
X_train, X_test, y_train, y_test = train_test_split(x, target, test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
clf.score(X_train, y_train)



test_result = pd.DataFrame()
test_result['Loan_ID']  =test["Loan_ID"]
test_result['Loan_Status'] = clf.predict(X_test)
test_result.to_csv("Sample.csv",index=False)















