from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.model_selection import train_test_split,GridSearchCV
import pandas as pd
import numpy as np

df_=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Turkcell Makine Öğrenmesi\\sınıflandırma\\diabetes.csv")
y=df_["Outcome"]
df=df_.drop(["Outcome"],axis=1)
x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.3,random_state=200)

catboost=CatBoostClassifier()
cb_params={
    "iterations":[200,500,100],
    "learning_rate":[0.01,0.001,0.1],
    "depth":[4,5,8]
}

catboostcv=GridSearchCV(catboost,cb_params,cv=5,n_jobs=-1,verbose=2)
catboostcv.fit(x_train,y_train)
iterations=catboostcv.best_params_["iterations"]
learning_rate=catboostcv.best_params_["learning_rate"]
depth=catboostcv.best_params_["depth"]

cb_tuned=CatBoostClassifier(depth=depth,iterations=iterations,learning_rate=learning_rate)
cb_tuned.fit(x_train,y_train)

predict=cb_tuned.predict(x_test)
acscore=accuracy_score(y_test,predict)
print(acscore)



from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.model_selection import train_test_split,GridSearchCV
import pandas as pd
import numpy as np

df_=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Turkcell Makine Öğrenmesi\\sınıflandırma\\diabetes.csv")
y=df_["Outcome"]
df=df_.drop(["Outcome"],axis=1)
x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.3,random_state=200)

catboost=CatBoostClassifier()
cb_params={
    "iterations":[200,500,100],
    "learning_rate":[0.01,0.001,0.1],
    "depth":[4,5,8]
}

catboostcv=GridSearchCV(catboost,cb_params,cv=5,n_jobs=-1,verbose=2)
catboostcv.fit(x_train,y_train)
iterations=catboostcv.best_params_["iterations"]
learning_rate=catboostcv.best_params_["learning_rate"]
depth=catboostcv.best_params_["depth"]

cb_tuned=CatBoostClassifier(depth=depth,iterations=iterations,learning_rate=learning_rate)
cb_tuned.fit(x_train,y_train)

predict=cb_tuned.predict(x_test)
acscore=accuracy_score(y_test,predict)
print(acscore)








