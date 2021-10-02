from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.shortcuts import render
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def welcome(request):
    return render(request, 'welcome.html',)
def result(request):
    path = r"student_info.csv"
    df = pd.read_csv(path)
    df2 = df.fillna(df.mean())
    #graph
    plt.scatter(x = df2.study_hours,y = df2.student_marks)
    plt.xlabel("STUDENTS STUDY HOURS")
    plt.ylabel("STUDENTS MARKS")
    plt.title("Scatter plot of Students study hours vs Students marks")
    #graph = plt.show()

    X = df2.drop("student_marks", axis="columns")
    y = df2.drop("study_hours", axis="columns")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=51)
    lr = LinearRegression()
    # test
    lr.fit(X_train, y_train)

    # accuracy
    acc = lr.score(X_test, y_test)
    # save the model
    joblib.dump(lr, "student_mark_predictor1.pkl")
    model = joblib.load('student_mark_predictor1.pkl')
    arrhours = request.POST['username']
    output = model.predict([[arrhours]])[0][0].round(2)
    fullmarks = int(request.POST['fullmarks'])
    cutoff = int(request.POST['cutoff'])
    percent2 = cutoff/fullmarks
    percent = percent2*100
    prob = output/percent
    if output > 100:
        output = 100
        if prob > 1:
            prob = 1
            return render(request, 'welcome.html', {'text': "You will get {}% percentage".format(output),
                                                    'text2': "Model accuracy is {}".format(round(acc, 2)),
                                                    'text3': "Your probability of selection in  this exam is {}".format(round(prob,2)),
                                                    'text4': "Exam full marks-{}  Cutoff marks-{}  Study hours-{}".format(
                                                        fullmarks, cutoff, arrhours)})
        else:
            return render(request, 'welcome.html', {'text': "You will get {}% percentage".format(output),
                                                    'text2': "Model accuracy is {}".format(round(acc, 2)),
                                                    'text3': "Your probability of selection in  this exam is {}".format(round(prob,2)),
                                                    'text4': "Exam full marks-{}  Cutoff marks-{}  Study hours-{}".format(
                                                        fullmarks, cutoff, arrhours)})
    else:
        if prob > 1:
            prob = 1
            return render(request, 'welcome.html', {'text': "You will get {}% percentage".format(output),
                                                    'text2': "Model accuracy is {}".format(round(acc, 2)),
                                                    'text3': "Your probability of selection in  this exam is {}".format(round(prob,2)),
                                                    'text4': "Exam full marks-{}  Cutoff marks-{}  Study hours-{}".format(
                                                        fullmarks, cutoff, arrhours)})
        else:
            return render(request, 'welcome.html', {'text': "You will get {}% percentage".format(output),
                                                    'text2': "Model accuracy is {}".format(round(acc, 2)),
                                                    'text3': "Your probability of selection in  this exam is {}".format(round(prob,2)),
                                                    'text4': "Exam full marks-{}  Cutoff marks-{}  Study hours-{}".format(
                                                        fullmarks, cutoff, arrhours)})

