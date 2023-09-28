import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 

#load data
data=pd.read_csv("HRA.csv", delimiter=",")

#since sales and salary are categorical variables encode them to numeric values
label_encoder = LabelEncoder()
data['sales'] = label_encoder.fit_transform(data['sales'])
data['salary'] = label_encoder.fit_transform(data['salary'])

X = data.drop('left', axis=1)
y = data['left']

# features and output
X = data.drop(['left'], axis=1)
y = data['left']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

decision_tree_model=DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)

def decision_tree():
    # Decision tree classifier
    decision_tree_model=DecisionTreeClassifier()
    decision_tree_model.fit(X_train, y_train)
    ypredict=decision_tree_model.predict(X_test)
    accuracy  = decision_tree_model.score(X_test,y_test)
    return accuracy
def random_forest():
    # Random forest classifier
    model=RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    ypredict=model.predict(X_test)
    print("Accuracy: ", model.score(X_test,y_test))
    cross_tab = pd.crosstab(y_test,ypredict)
    cross_tab
def KNN():
    # K Nearest Neighbors
    model=KNeighborsClassifier()
    model.fit(X_train, y_train)
    ypredict=model.predict(X_test)
    print("Accuracy: ", model.score(X_test,y_test))
    cross_tab = pd.crosstab(y_test,ypredict)
    cross_tab
def logistic():
    #logistic regression
    model = LogisticRegression(random_state=0)
    model.fit(X_train, y_train)
    ypredict=model.predict(X_test)
    print("Accuracy: ", model.score(X_test,y_test))
    cross_tab = pd.crosstab(y_test,ypredict)
    cross_tab


def predict(number_project, average_montly_hours, time_spend_company, Work_accident, left, promotion_last_5years, sales, salary):
    sales = label_encoder.fit_transform([sales])
    salary = label_encoder.fit_transform([salary])
    prediction = decision_tree_model.predict([[number_project, average_montly_hours, time_spend_company, Work_accident, left, promotion_last_5years, sales, salary]])
    return prediction

def predict_left():
    st.title("Employee details")

    with st.form("my_form"):
        
        number_project = st.number_input("number_project")
        left=st.number_input("left")
        average_montly_hours = st.number_input("average_montly_hours")
        time_spend_company = st.number_input("time_spend_company")
        Work_accident = st.number_input("Work_accident")
        promotion_last_5years = st.number_input("promotion_last_5years")
        sales = st.selectbox("Select your department", ["sales", "accounting", "hr", "technical", "support", "management", "IT", "product_mng", "marketing", "RandD"])
        salary = st.selectbox("Select your salary", ["low", "medium", "high"])
        submitted_user = st.form_submit_button("Submit")
    
    if submitted_user:
        prediction = predict(number_project, average_montly_hours, time_spend_company, Work_accident, left, promotion_last_5years, sales, salary)
        if prediction == 0:
            st.write("Employee is not likely to leave")
        else:
            st.write("Employee is likely to leave")
    

def models():
    st.write("## Models performance")
    st.write("### Decision tree")
    st.write(decision_tree())
    st.write("### Random forest")
    st.write(random_forest())
    st.write("### KNN")
    st.write(KNN())
    st.write("### Logistic regression")
    st.write(logistic())


with st.sidebar:
    st.write("# Select")

menu = ["Get prediction", "Models performance"]
choice = st.sidebar.selectbox("", menu)

if choice == "Get prediction":
    predict_left()
elif choice == "Models performance":
    models()