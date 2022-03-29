import os
import sys
from PyQt5.QtWidgets import *
from os.path import dirname, realpath, join
from PyQt5.uic import loadUiType
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, mean_squared_log_error
import numpy as np
from collections.abc import Iterable
from pandas_model import PandasModel
from sklearn.ensemble import RandomForestClassifier

scriptDir = dirname(realpath(__file__))
From_Main, _ = loadUiType(join(dirname(__file__), "Main.ui"))#преообразовыввает ui в py

class MainWindow(QWidget, From_Main):
    def __init__(self):
        super(MainWindow, self).__init__()
        QWidget.__init__(self)
        self.setupUi(self)

        self.activation_ComboBox.addItems(["relu", "sigmoid", "softmax"])#селект
        self.activation_ComboBox_2.addItems(["relu", "sigmoid", "softmax"])#селект
        self.activation_ComboBox_3.addItems(["relu", "sigmoid", "softmax"])#селект

        
        self.openFileButton.clicked.connect(self.OpenFile)

        self.trainingButton.clicked.connect(self.Training)
        self.trainingButton_2.clicked.connect(self.modelLoss)
        self.trainingButton_3.clicked.connect(self.modelAccuracy)
        self.pushButton.clicked.connect(self.saveResult)    

    def OpenFile(self):
        # Чтение SCV файла
        path = QFileDialog.getOpenFileName(
            self, 'Open CSV', os.getenv('HOME'), 'CSV(*.csv)')[0]
        self.df = pd.get_dummies(pd.read_csv(path))
        self.df = pd.get_dummies(self.df)  
        self.populateTableView()

    def populateTableView(self):
        model = PandasModel(self.df)
        self.tableView.setModel(model)
        self.trainColumn.addItems(self.df.columns)

    def Training(self):
        if self.trainingMethod.currentText() == "RandomForest":
            self.X = pd.get_dummies(self.df.drop([self.trainColumn.currentText()], axis=1))
            self.Y = self.df[self.trainColumn.currentText()]
            self.X_train, self.X_val_and_test, self.Y_train, self.Y_val_and_test = train_test_split(self.X,self.Y,test_size = self.testSizeValue.value()/100,train_size = self.trainDataSize.value()/100)
            model = RandomForestClassifier(n_estimators = self.trees.value(), max_depth=self.max_depth.value(), random_state=30)
            model.fit(self.X_train, self.Y_train)
            self.Y_pred = model.predict(self.X_val_and_test)
            self.Y_pred = self.Y_pred.flatten()
            print("RF pref =")
            print(self.Y_pred)
        
            rowPosition = self.predictedTable.rowCount()
            self.predictedTable.setHorizontalHeaderLabels(["Predicted", self.trainColumn.currentText()])
            for idx, val in enumerate(self.Y_pred):
                self.predictedTable.insertRow(0+idx)
                self.predictedTable.setItem(0+idx , 0 ,QTableWidgetItem(f'{val}'))
            for idx, val in enumerate(self.Y_val_and_test):
                self.predictedTable.insertRow(0+idx)
                self.predictedTable.setItem(0+idx , 1 ,QTableWidgetItem(f'{val}'))
        else: self.notRandomForest()
    
    def notRandomForest(self): 
        if self.trainingMethod.currentText() == 'ADAM':
            tf.keras.optimizers.Adam(
                learning_rate=self.learningRate_DoubleSpinBox.value(),
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=False,
                name='Adam'
            )
        elif self.trainingMethod.currentText() == 'ADADELTA':
            tf.keras.optimizers.Adadelta(
                learning_rate=self.learningRate_DoubleSpinBox.value(),
                rho=0.95,
                epsilon=1e-07,
                name='Adadelta'
            )
        elif self.trainingMethod.currentText() == 'SGD':
            tf.keras.optimizers.SGD(
                learning_rate=self.learningRate_DoubleSpinBox.value(),
                momentum=0.0,
                nesterov=False,
                name='SGD'
            )
        elif self.trainingMethod.currentText() == 'RMSprop':
            tf.keras.optimizers.RMSprop(
                learning_rate=self.learningRate_DoubleSpinBox.value(),
                rho=0.9,
                momentum=0.0,
                epsilon=1e-07,
                centered=False,
                name='RMSprop'
            )

        self.X = pd.get_dummies(self.df.drop([self.trainColumn.currentText()], axis=1))
        self.Y = self.df[self.trainColumn.currentText()]

        self.X_train, self.X_val_and_test, self.Y_train, self.Y_val_and_test = train_test_split(
            self.X,
            self.Y,
            test_size = self.testSizeValue.value()/100,
            train_size = self.trainDataSize.value()/100
        )

        self.X_val, self.X_test, self.Y_val, self.Y_test = train_test_split(
            self.X_val_and_test,
            self.Y_val_and_test,
            test_size = self.testValidateSize.value()/100
        )

        self.model = Sequential()

        self.model.add(
            Dense(
                units=self.neuronsNumberFirstLayer_SpinBox.value(),
                activation = self.activation_ComboBox.currentText(),
                input_dim = len(self.X_train.columns)
            )
        )

        self.model.add(Dense(
                units=self.neuronsNumberSecondLayer_SpinBox.value(),
                activation = self.activation_ComboBox_2.currentText()
            ))

        self.model.add(Dense(
                units=self.neuronsNumberOutputLayer_SpinBox.value(),
                activation = self.activation_ComboBox_3.currentText()))

        self.model.compile(
            optimizer = self.trainingMethod.currentText(),
            loss = 'binary_crossentropy',
            metrics = 'accuracy'
        )

        self.trainHist = self.model.fit(
            self.X_train,
            self.Y_train,
            batch_size=self.batchSize_SpinBox.value(),
            epochs=self.epochs_SpinBox.value(),
            validation_data = (self.X_val, self.Y_val)
        )

        self.model.evaluate(self.X_train, self.Y_train)
        self.Y_pred = self.model.predict(self.X_val_and_test)
        self.Y_pred = self.Y_pred.flatten()
    
        self.predictedTable.setRowCount(self.Y_pred.size + 1)
        self.predictedTable.setColumnCount(2)  
        self.predictedTable.setItem(0 , 0 ,QTableWidgetItem(f'Predict'))
        self.predictedTable.setItem(0 , 1 ,QTableWidgetItem(f'{self.trainColumn.currentText()}'))
        for idx, val in enumerate(self.Y_pred):
            self.predictedTable.setItem(1+idx , 0 ,QTableWidgetItem(f'{round(float(val), 2)}'))
        for idx, val in enumerate(self.Y_val_and_test):
            self.predictedTable.setItem(1+idx , 1 ,QTableWidgetItem(f'{val}'))
        # newDf = pd.DataFrame(self.Y_pred),
        # predictModel = PandasModel(newDf)



        self.MSE.setText(f'{round(mean_squared_error(self.Y_val_and_test, self.Y_pred), 2) if self.neuronsNumberOutputLayer_SpinBox.value() == 1 else 0}')
        self.RMSE.setText(f'{round(np.sqrt(mean_squared_error(self.Y_val_and_test, self.Y_pred)), 2) if self.neuronsNumberOutputLayer_SpinBox.value() == 1 else 0}')
        self.MAE.setText(f'{round(mean_absolute_error(self.Y_val_and_test, self.Y_pred), 2) if self.neuronsNumberOutputLayer_SpinBox.value() == 1 else 0}')
        self.maxError.setText(f'{round(max_error(self.Y_val_and_test, self.Y_pred), 2)}')
        self.msle.setText(f'{round(mean_squared_log_error(self.Y_val_and_test, self.Y_pred), 2)}')

        print(
            self.testSizeValue.value()/100,'\n',
            self.testValidateSize.value()/100,'\n',
            self.trainingMethod,'\n',
            self.epochs_SpinBox.value(),'\n',
            self.activation_ComboBox.currentText(),'\n',
            self.learningRate_DoubleSpinBox.value(),'\n',
            self.neuronsNumberFirstLayer_SpinBox.value(),'\n',
            self.neuronsNumberSecondLayer_SpinBox.value(),'\n',
            self.neuronsNumberOutputLayer_SpinBox.value(),'\n',
            self.batchSize_SpinBox.value(),'\n'
        )


    def modelLoss(self):
        plt.plot(self.trainHist.history['loss'])
        plt.plot(self.trainHist.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc = 'upper right')
        self.graph_1 = QPixmap(plt.show())

    def modelAccuracy(self):
        plt.plot(self.trainHist.history['accuracy'])
        plt.plot(self.trainHist.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc = 'lower right')
        self.graph_1 = QPixmap(plt.show())

    def saveResult(self):
        def flatten(l):
            for el in l:
                if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
                    yield from flatten(el)
                else:
                    yield el
        
        self.res = list(flatten(self.Y_pred))
        self.df['Predict'] = pd.Series(self.res)

        self.model.save('olya.h5')
        del self.model

app = QApplication(sys.argv)
sheet = MainWindow()
sheet.show()
sys.exit(app.exec_())
