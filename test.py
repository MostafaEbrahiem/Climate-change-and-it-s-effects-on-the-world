import matplotlib.pyplot as plt
import csv
from tkinter import *
import tkinter.messagebox
import numpy as np
from PIL import ImageTk,Image
from tkinter import filedialog
from pickle import dump
import os
import tflearn
import gc
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import  conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import  time
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def plot1():
    x = []
    y = []

    with open('carbonmonitor-global_datas_2021-10-03.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        for row in plots:
            if row[3] == "value" or row[0] == "country":
                0
            else:
                x.append(row[0])
                y.append(float(row[3]))

    plt.bar(x, y, color='g', width=0.72, label="value")
    plt.xlabel('Names')
    plt.ylabel('value')
    plt.title('carbon')
    plt.legend()
    plt.show()


def plot2():
    x = []
    y = []

    with open('GlobalLandTemperaturesByCity.csv',encoding="utf8" ) as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        for row in plots:
            if row[1] == "AverageTemperature" or row[4] == "Country" or row[1] =="":
                0
            else:
                x.append(row[4])
                y.append(float(row[1]))

    plt.bar(x, y, color='g', width=0.72, label="AverageTemperature")
    plt.xlabel('Names')
    plt.ylabel('value')
    plt.title('AverageTemperature')
    plt.legend()
    plt.show()

def predict():
    Movies_training= pd.read_csv('carbonmonitor-global_datas_2021-10-03.csv')

    X = Movies_training['country']  # X for single linear reg
    t_x = []
    for i in X:
        t_x.append(i)

    x_label_encoder = LabelEncoder()
    t_x = x_label_encoder.fit_transform(t_x)
    Movies_training['country'] = t_x

    X = Movies_training['country']  # X for single linear reg
    Y = Movies_training['value']  # Label FOR SIGLE LINEAR
    # Split the data to training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1, shuffle=True)

    linear_reg_start_time = time.time()
    cls = linear_model.LinearRegression()
    X_train = np.expand_dims(X_train, axis=1)
    y_train = np.expand_dims(y_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    y_test = np.expand_dims(y_test, axis=1)

    # train
    cls.fit(X_train, y_train)
    dump(cls, open('linear_reg.pkl', 'wb'))
    #cls = load(open('linear_reg.pkl', 'rb'))
    prediction = cls.predict(X_test)
    # prediction = cls.predict(X_test)
    linear_reg_end_time = time.time()
    plt.scatter(X_train, y_train)
    plt.xlabel('(Brazil,China,UK,France,Germany,India,italy,Japan,ROW,Russia,Spain,US,World)', fontsize=20)
    plt.ylabel('value', fontsize=20)
    plt.plot(X_test, prediction, color='red', linewidth=3)
    plt.show()
    print('linear rag time = ', linear_reg_end_time - linear_reg_start_time)
    # //////////print('Co-efficient of SINGLE linear regression',cls.coef,MSE,ACCURACY_)////////////

    # print('coef of Single regression model', cls.coef_)
    # print('Intercept of Single regression model', cls.intercept_)
    print('Mean Square Error in Single linear regression', metrics.mean_squared_error(np.asarray(y_test), prediction))
    print('Accuracy of Single regression', cls.score(y_test, prediction))
    print('r2 score of Single regression', metrics.r2_score(y_test, prediction))
    true_rate_value = np.asarray(y_test)[0]
    predicted_rate_value = prediction[0]
    print('True value for the first movie in the test is : ' + str(true_rate_value))
    print('Predicted value for the first movie in the test set is : ' + str(predicted_rate_value))


def main():

    Upload_butt = Button(text="   carbon percentage   ", command=plot1)
    Upload_butt.place(x=350, y=600)

    Upload_butt = Button(text="   Warming percentage   ", command=plot2)
    Upload_butt.place(x=550, y=600)

    Upload_butt = Button(text="   predict carbon percentage   ", command=predict)
    Upload_butt.place(x=750, y=600)

    img = Image.open("1.jpg")
    img = img.resize((1200, 650), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor=NW, image=img)


    form.mainloop()




if __name__ == '__main__':
    form = Tk()
    canvas = Canvas(form, width=1200, height=650)
    canvas.pack()
    main()