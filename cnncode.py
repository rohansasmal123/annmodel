import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')

dataset.head()

y = dataset['Exited']

X = dataset[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary']]

Geography = pd.get_dummies(dataset['Geography'],drop_first=True)

Gender = pd.get_dummies(dataset['Gender'],drop_first=True)

X = pd.concat([X,Geography, Gender], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

from keras.optimizers import Adam

from keras.models import Sequential

from keras.layers import Dense

model = Sequential()

model.add(Dense(units=10, input_dim=11, activation='relu' ))

model.add(Dense(units=10, activation='relu'))

model.add(Dense(units=8, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

train_model = model.fit(X_train, y_train, epochs=30)

y_pred = model.predict(X_test)

text = train_model.history
accuracy = text['accuracy'][-1] * 100
model.save("model.h5")


f = open('accuracy.txt','w')
f.write("{0}".format(accuracy))
f.close()
