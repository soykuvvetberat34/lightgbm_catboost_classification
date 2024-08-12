from lightgbm import LGBMClassifier
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score#doğruluk oranı
from sklearn.model_selection import train_test_split,GridSearchCV
import pandas as pd
import numpy as np

df_=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Turkcell Makine Öğrenmesi\\sınıflandırma\\diabetes.csv")
y=df_["Outcome"]#bağımlı değişken
df=df_.drop(["Outcome"],axis=1)#bağımsız değişkenler
x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.3,random_state=200)

lgbmc=LGBMClassifier()
lgbmc_params={#lgbm hiperparametrelerinden bazılarını alarak bunlara en uygun değerleri atayacağız
    "n_estimators":[100,200,500],
    "learning_rate":[0.001,0.01,0.1],
    "max_depth":[1,2,3,5,8]
}
#hiper parametrelerin en iyi değerlerini bulma
lgbmcv=GridSearchCV(lgbmc,lgbmc_params,cv=5,n_jobs=-1,verbose=2)
lgbmcv.fit(x_train,y_train)
n_estimators=lgbmcv.best_params_["n_estimators"]
learning_rate=lgbmcv.best_params_["learning_rate"]
max_depth=lgbmcv.best_params_["max_depth"]

lgbm_tuned=LGBMClassifier(max_depth=max_depth,learning_rate=learning_rate,n_estimators=n_estimators)
lgbm_tuned.fit(x_train,y_train)

#tahminleri bul
predict=lgbm_tuned.predict(x_test)
acscore=accuracy_score(y_test,predict)
print(acscore)



