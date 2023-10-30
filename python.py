# imort libs

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# load data

data = pd.read_csv("Mall_Customers_sep2022.csv")
print(data)

# understand data

print(data.shape) 
res = data.isnull().sum()
print(res)

# handling missing values - remove rows with NaNs
data = data.dropna()

# features 

features = data[["Annual_Income", "Spending_Score"]]

# feature scaling

mms = MinMaxScaler()
nfeatures = mms.fit_transform(features)
print(features)
print(nfeatures)

# model 

model = KMeans(n_clusters=3, random_state=123)

# fit_predict

res = model.fit_predict(nfeatures)
data["clusters"] = res
print(data)

# seperate clusters
data0 = data[data.clusters==0]
data1 = data[data.clusters==1]
data2 = data[data.clusters==2]
data3 = data[data.clusters==3]
data4 = data[data.clusters==4]


plt.scatter(data0["Annual_Income"], data0["Spending_Score"], color="red", label="0")
plt.scatter(data1["Annual_Income"], data1["Spending_Score"], color="yellow", label="1")
plt.scatter(data2["Annual_Income"], data2["Spending_Score"], color="green", label="2")
plt.scatter(data3["Annual_Income"], data3["Spending_Score"], color="blue", label="3")
plt.scatter(data4["Annual_Income"], data4["Spending_Score"], color="orange", label="4")
plt.legend()
plt.show()

# predict

x = float(input("enter Annual Income "))
y = float(input("enter Spending Score "))
ipdata = [[x, y]]
nd = mms.transform(ipdata)
ans = model.predict(nd)

if ans == 0:
	print("You Should Buy Life Insurance ")
elif ans == 1:
	print("You Should Buy A New Car ")
elif ans == 2:
	print("You Should Buy Mutual Funds ")
elif ans == 3:
	print("You Should Buy New Phone ")
else:
	print("You Should New Watch ")








Mall_Customers_sep2022.csv

CustomerID,Gender,Age,Annual_Income,Spending_Score
1,Male,19,15,39
2,Male,21,15,81
3,Female,21,16,91
...

pip install pandas scikit-learn matplotlib
