
# coding: utf-8


"""
The traget of this file is to study the Process MD platform based on Partial Least Squares (PLS).

Modified at  2017.10.Aug

"""

# Author: Carlos JI <jic4@airproducts.com>

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import scale,Imputer
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error,r2_score

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Scatter, Figure, Layout
import plotly.graph_objs as go
init_notebook_mode(connected=True)

import sys
import time

class PLSModel():
    
    def __init__(self,filepath = None,y = 'Y', date = 'Date'):
        
        self.filepath = None
        self.y = y
        self.date = date
        self.Filter = None
        self.X = None
        self.Y = None
        self.Date = None
        self.k = None
        self.t = None
        self.Summary = None
        self.Formula = None
        self.Tags = None
        self.Source = None
        self.PLSData = None
        self.PC = None
        self.Test = None
    
    def load(self,filepath,y='Y',date=None,Filter=None):
        
        self.filepath = filepath
        self.y = y
        self.date = date
        self.Filter = Filter        
        self._load(self.filepath,self.y,self.date)
    
    def _load(self,filepath,y,date):
        
        self.Source = pd.read_csv(self.filepath)
                        
        if self.date != None:
            self.Date = list(map(lambda x: datetime.strptime(x,'%m/%d/%Y'),self.Source[self.date]))
            self.Source[self.date] = self.Date
        else:
            self.Date = []
        
        return self
  
    def Calculate(self,PC=None,CV='Y',Filter= None,index = None):   
        
        if index is None:
            
            index = self.Source.index
        
        if Filter != None:
            
            index = self.Source.loc[self.Source.index,:][self.Source.loc[:,self.Filter]<120].index
            index = list(set(c.Source.index).difference(set(index).union(set(index+1))))
        if (PC is None) & (self.PC != None):
            
            PC = self.PC
            
        self.Summary,self.Formula,self.Tags = self._PLS(PC = PC,index = index)

    def _PLS(self,PC,index,CV='Y'):
        
        start_time = time.time()
        
        dSummary = pd.DataFrame(np.zeros((10,5),dtype = int),columns=['R2','R2 Cum','Q2','Q2 Cum','RMSE'])
        dF = self.Source.loc[index].drop(self.Filter,axis=1).dropna()
        self.Y = dF[self.y].astype('float64')
        self.X = dF.drop([self.y,self.date],axis=1).astype('float64')
        l = len(self.X.columns)
       
        if PC is None:
            n = l
        elif PC < l:
            n = PC + 1
        else:    
            n = l
    
        for i in np.arange(1, n):
            
            pls = PLSRegression(n_components=i)
            pls.fit(scale(self.X),self.Y)
            self.k = (pls.coef_.T/np.array(self.X.std(axis=0, ddof=1))).T
            self.t = self.Y.mean(axis=0)-np.sum(np.array(self.X.mean(axis=0)/self.X.std(axis=0, ddof=1))*pls.coef_.T)

            
            if CV == 'Y':
                Y_true=self.Y.values.reshape((-1, 1))
                Y_pred = pls.predict(scale(self.X)).reshape((-1, 1))
                numerator = ((Y_true - Y_pred) ** 2).sum(axis=0, dtype=np.float64)
                denominator = ((Y_true - np.average(Y_true, axis=0)) ** 2).sum(axis=0,dtype=np.float64)
                kf = KFold(n_splits=len(self.X), shuffle=True, random_state=2)
                PRESS = 0
                for train_index, test_index in kf.split(self.X):
                    X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                    Y_train, Y_test = self.Y.iloc[train_index], self.Y.iloc[test_index]
                    pls1 = PLSRegression(n_components=i)
                    pls1.fit(scale(X_train), Y_train)
                    kt = (pls1.coef_.T/np.array(X_train.std(axis=0, ddof=1))).T
                    tt = Y_train.mean(axis=0)-np.sum(np.array(X_train.mean(axis=0)/X_train.std(axis=0, ddof=1))*pls1.coef_.T)
                    PRESS += ((Y_test.values.reshape((-1,1))-(np.dot(np.array(X_test),kt)+tt))**2).sum(axis=0, dtype=np.float64)
                q2 = round(np.average(1 - PRESS/denominator),3)
                
            dSummary.loc[i,'R2'] = round(r2_score(self.Y, pls.predict(scale(self.X)))-dSummary.loc[i-1,'R2 Cum'],3)
            dSummary.loc[i,'R2 Cum'] = round(r2_score(self.Y, pls.predict(scale(self.X))),3)
            dSummary.loc[i,'Q2'] = round(q2-dSummary.loc[i-1,'Q2 Cum'],3)
            dSummary.loc[i,'Q2 Cum'] = round(q2,3)
            dSummary.loc[i,'RMSE'] = round(mean_squared_error(self.Y, pls.predict(scale(self.X)))**0.5,2)
            
            if (dSummary.loc[i,'R2']<0.01) & ((PC is None) or (PC > l)):
                PC = i - 1
                break
        
        pls = PLSRegression(n_components=PC)
        pls.fit(scale(self.X), self.Y)
        self.k = (pls.coef_.T/np.array(self.X.std(axis=0, ddof=1))).T
        self.t = self.Y.mean(axis=0)-np.sum(np.array(self.X.mean(axis=0)/self.X.std(axis=0, ddof=1))*pls.coef_.T)
        
        dFor = pd.DataFrame(self.k.T,columns=list(self.X.columns))
        dFor['Intercept'] = self.t
        dPLSCoe = pd.DataFrame(pls.coef_.T,columns=list(self.X.columns))

        self.PC = PC
        self.PLSData = pd.DataFrame(columns = ['Date','Actual','Predicted','Residual','SPEY','SPEX','ResidualX'])
        self.PLSData.Date = self.Source.loc[dF.index,self.date]
        self.PLSData.Actual = self.Y
        self.PLSData.Predicted = (np.dot(np.array(self.X),self.k)+self.t)
        self.PLSData.Residual = np.array(self.Y).reshape(-1,1) - (np.dot(np.array(self.X),self.k)+self.t)
        self.PLSData.SPEY = (self.PLSData.Actual-self.PLSData.Predicted)**2
        self.PLSData.SPEX = 0
        self.PLSData.ResidualX = 0
        
        dTags = pd.DataFrame(index=list(self.X.columns),columns=['Residual Mean','Residual Std','Missing Data Ratio'])
        for i in range(0,l):
            dF = self.X.assign(Y=self.Y).drop(self.X.columns[i],axis=1)
            pls2 = PLSRegression(n_components=PC)
            pls2.fit(scale(dF), self.X.iloc[:,i])
            kf = (pls2.coef_.T/np.array(dF.std(axis=0, ddof=1))).T
            tf = self.X.iloc[:,i].mean(axis=0)-np.sum(np.array(dF.mean(axis=0)/dF.std(axis=0, ddof=1))*pls2.coef_.T)
            Residual = np.array(self.X.iloc[:,i]).reshape(-1,1)-(np.dot(np.array(dF),kf)+tf)
            self.PLSData.SPEX =  np.array(self.PLSData.SPEX) + Residual**2
            dTags.iloc[i,0] = round(Residual.mean(),4)
            dTags.iloc[i,1] = round(Residual.std(),2)
            dTags.iloc[i,2] = round(1-len(self.Source.loc[index,self.X.columns[i]].dropna())/len(self.Source.loc[index,self.X.columns[i]]),2)

        dTags.loc[:,'PLS Coefficients'] = pls.coef_/4
        self.PLSData.SPEX = self.PLSData.SPEX - self.PLSData.SPEX.min()
        print("--- Running time: %s seconds ---" % round(time.time() - start_time,2))
        
        return dSummary.iloc[1:PC+1,:],dFor,dTags
    
    def Report(self):
        
        print('Summary:')
        print(self.Summary)
        print('\n')
        print('Tags:')
        print(self.Tags.loc[:,('Residual Mean','Residual Std')])
        print('\n')
        print('Missing data ratio')
        print(self.Tags[self.Tags.loc[:,'Missing Data Ratio']!=0].loc[:,'Missing Data Ratio'])

    
    def Figure(self):
        traceAct = go.Scatter(
            x = self.PLSData.Date,
            y = self.PLSData.Actual,
            mode = 'lines',
            name = 'Actual Y',
            connectgaps = True
        )
        tracePre = go.Scatter(
            x = self.PLSData.Date,
            y = self.PLSData.Predicted,
            mode = 'lines',
            name = 'Predicted Y',
            connectgaps = True
        )

        plot(go.Figure(
            data=[traceAct,tracePre],
            layout = go.Layout(title='Actual Y vs Predicted Y')
            ),
            filename='Actual Y vs Predicted Y'
        )
        
       
        traceRes = go.Scatter(
            x = self.PLSData.Date,
            y = self.PLSData.Residual,
            mode = 'lines',
            name = 'Residual Y',
            connectgaps = True
        )
        
        plot(go.Figure(
            data=[traceRes],
            layout = go.Layout(title='Residual Y')
            ),
            filename='Residual Y.html'
        )

        traceSPEX = go.Scatter(
            x = self.PLSData.Date,
            y = self.PLSData.SPEX,
            mode = 'lines',
            name = 'Process Data Error',
            connectgaps = True
        )        
        
        traceSPEY = go.Scatter(
            x = self.PLSData.Date,
            y = self.PLSData.SPEY,
            mode = 'lines',
            name = 'Quality Data Error',
            connectgaps = True
        )
        
        plot(go.Figure(
            data=[traceSPEX],
            layout = go.Layout(title='Process Data Error')
            ),
            filename='Process Data Error.html'            
        )
        
        plot(go.Figure(
            data=[traceSPEY],
            layout = go.Layout(title='Quality Data Error')
            ),
            filename='Quality Data Error.html'         
        )
        
        plot(go.Figure(
            data=[go.Bar(x=self.Tags.index,y=self.Tags.loc[:,'PLS Coefficients'])],
            layout = go.Layout(title='PLS Coefficients')
            ),
            filename='PLS Coefficients.html'                 
        )
        
    
    def Expression(self,Tag = None):
       
        dT = self.Formula.drop('Intercept',axis=1)
        
        l = len(dT.columns)
        
        sFor = ''
        
        if Tag is None: 
            for i in range(0,l):
                if dT.iloc[0,i] < 0:
                    sFor = sFor + dT.iloc[0,i].astype('str') + '*{' + dT.columns[i] + '}' 
                else:
                    sFor = sFor + '+' + dT.iloc[0,i].astype('str') + '*{' + dT.columns[i] + '}' 

            if self.Formula.loc[0,'Intercept'] < 0:
                sFor = sFor + self.Formula.loc[0,'Intercept'].astype('str')
            else:
                sFor = sFor + '+' + self.Formula.loc[0,'Intercept'].astype('str')
        else:
            for i in range(0,l):
                if dT.iloc[0,i] < 0:
                    sFor = sFor + dT.iloc[0,i].astype('str') + '*' + Tag[i]
                else:
                    sFor = sFor + '+' + dT.iloc[0,i].astype('str') + '*' + Tag[i]
            sFor = sFor + '+' + self.Formula.loc[0,'Intercept'].astype('str')
        
        return sFor
    
    def Optimize(self,item = 'Residual',threshold = 5):
        
        item = item.lower()
        print(item)
        
        if (item == 'residual'):
            
            index = self.PLSData[abs(self.PLSData.Residual)<threshold].index
        
        elif (item == 'actual'):
            
            index = self.PLSData[abs(self.PLSData.Actual)<threshold].index
            
        elif (item == 'spey'):
            
            index = self.PLSData[abs(self.PLSData.SPEY)<threshold].index
        
        elif (item == 'spex'):
            
            index = self.PLSData[abs(self.PLSData.SPEX)<threshold].index
        
        else:
            
            print('Error of item')
            sys.exit
               
        self.Calculate(PC=self.PC,index=index)

Tags=['((CAPP("hf", "pt", UOMCONV({YL4_PI1840.PV}, "bar_g", "psia"), UOMCONV({YL4_TI2057.PV}, "C", "F"), ARRAY(0.78, 0.21, 0.01), ARRAY("N2", "O2", "AR"), 19)-CAPP("hf", "pt", UOMCONV({YL4_PI1840.PV}, "bar_g", "psia"), UOMCONV({YL4_TI2021.PV}, "C", "F"), ARRAY(0.78, 0.21, 0.01), ARRAY("N2", "O2", "AR"), 19))*(UOMCONV({YL4_FIC1840.PV}, "m3/hr", "ft3/hr"))*CAPP("denv", "pt", 14.7, 32, ARRAY(0.78, 0.21, 0.01), ARRAY("N2", "O2", "AR"), 19))/CAPP("mwv", "pt", 14.7, 32, ARRAY(0.78, 0.21, 0.01), ARRAY("N2", "O2", "AR"), 19)/((({YL4_TI2057.PV}-({YL4_TI2013.PV}+{YL4_TI2032.PV})/2)-({YL4_TI2021.PV}-({YL4_TI2029.PV}+{YL4_TI2110.PV})/2))/(ln({YL4_TI2057.PV}-({YL4_TI2013.PV}+{YL4_TI2032.PV})/2)-ln({YL4_TI2021.PV}-({YL4_TI2029.PV}+{YL4_TI2110.PV})/2)))',
  '(((CAPP("hf", "pt", UOMCONV({YL4_PIC1303.PV}, "bar_g", "psia"), UOMCONV({YL4_TI1322.PV}, "C", "F"), ARRAY(0.78, 0.21, 0.01), ARRAY("N2", "O2", "AR"), 19)-CAPP("hf", "pt", UOMCONV({YL4_PIC1303.PV}, "bar_g", "psia"), UOMCONV({YL4_TI2070.PV}, "C", "F"), ARRAY(0.78, 0.21, 0.01), ARRAY("N2", "O2", "AR"), 19))*UOMCONV({YL4_FI1325_CALC.PV}, "m3/hr", "ft3/hr")) + ((CAPP("hf", "pt", UOMCONV({YL4_PIC3706.PV}, "bar_g", "psia"), UOMCONV({YL4_TI3706.PV}, "C", "F"), ARRAY(0.78, 0.21, 0.01), ARRAY("N2", "O2", "AR"), 19)-CAPP("hf", "pt", UOMCONV({YL4_PIC3706.PV}, "bar_g", "psia"), UOMCONV({YL4_TI3712.PV}, "C", "F"), ARRAY(0.78, 0.21, 0.01), ARRAY("N2", "O2", "AR"), 19))*UOMCONV({YL4_FIC3712.PV}, "m3/hr", "ft3/hr")) + ((CAPP("hf", "pt", UOMCONV({YL4_PIC2606.PV}, "bar_g", "psia"), UOMCONV({YL4_TI2606.PV}, "C", "F"), ARRAY(0.78, 0.21, 0.01), ARRAY("N2", "O2", "AR"), 19)-CAPP("hf", "pt", UOMCONV({YL4_PIC2606.PV}, "bar_g", "psia"), UOMCONV({YL4_TIC2612-1.PV}, "C", "F"), ARRAY(0.78, 0.21, 0.01), ARRAY("N2", "O2", "AR"), 19))*UOMCONV({YL4_FIC2612.PV}, "m3/hr", "ft3/hr")) ) *(CAPP("denv", "pt", 14.7, 32, ARRAY(0.78, 0.21, 0.01), ARRAY("N2", "O2", "AR"), 19) / CAPP("mwv", "pt", 14.7, 32, ARRAY(0.78, 0.21, 0.01), ARRAY("N2", "O2", "AR"), 19)) / (((({YL4_TI1322.PV}+{YL4_TI3706.PV}+{YL4_TI2606.PV})/3-{YL4_TI2020-1.PV})-(({YL4_TI2070.PV}+{YL4_TI3712.PV}+{YL4_FIC2612.PV})/3-{YL4_TI2010.PV})) / (ln(({YL4_TI1322.PV}+{YL4_TI3706.PV}+{YL4_TI2606.PV})/3-{YL4_TI2020-1.PV})-ln(({YL4_TI2070.PV}+{YL4_TI3712.PV}+{YL4_FIC2612.PV})/3-{YL4_TI2010.PV})))',
   '((3.5746*UOMCONV({YL4_TI1108.PV}, "C", "K")*Ln(UOMCONV({YL4_PI1100.PV}, "bar_g", "psia")/UOMCONV({YL4_PI1109.PV}, "bar_g", "psia")))*(UOMCONV({YL4_FIC1840.PV}*1000, "m3/hr", "ft3/hr")*CAPP("denv", "pt", 14.7, 32, ARRAY(0.78, 0.21, 0.01), ARRAY("N2", "O2", "AR"), 19)/CAPP("mwv", "pt", 14.7, 32, ARRAY(0.78, 0.21, 0.01), ARRAY("N2", "O2", "AR"), 19))+(3.5746*UOMCONV({YL4_TI1301-2.PV}, "C", "K")*Ln(UOMCONV({YL4_PIC1304.PV}, "bar_g", "psia")/UOMCONV({YL4_PI1321-2.PV}, "bar_g", "psia")))*(UOMCONV({YL4_FIC1319.PV}, "m3/hr", "ft3/hr")*CAPP("denv", "pt", 14.7, 32, ARRAY(0.78, 0.21, 0.01), ARRAY("N2", "O2", "AR"), 19)/CAPP("mwv", "pt", 14.7, 32, ARRAY(0.78, 0.21, 0.01), ARRAY("N2", "O2", "AR"), 19))+(3.5746*UOMCONV({YL4_TI1303-1.PV}, "C", "K")*Ln(UOMCONV({YL4_PIC1303.PV}, "bar_g", "psia")/UOMCONV({YL4_PI1308-2.PV}, "bar_g", "psia")))*(UOMCONV({YL4_FIC1304.PV}, "m3/hr", "ft3/hr")*CAPP("denv", "pt", 14.7, 32, ARRAY(0.78, 0.21, 0.01), ARRAY("N2", "O2", "AR"), 19)/CAPP("mwv", "pt", 14.7, 32, ARRAY(0.78, 0.21, 0.01), ARRAY("N2", "O2", "AR"), 19)))*100/((CAPP("hf", "pt", UOMCONV({YL4_PI8640.PV}, "bar_g", "psia"), UOMCONV({YL4_TI8630A.PV}, "C", "F"), ARRAY(1), ARRAY("WA"), 19)-CAPP("hf", "pv", UOMCONV({YL4_PIC8665-1.PV}, "bar_a", "psia"), 0.9, ARRAY(1), ARRAY("WA"), 19))*(UOMCONV({YL4_FI9902.PV}, "t/hr", "lb/hr")/18))',
   '{YL4_FI1325_CALC.PV}/{YL4_FIC1840.PV}',
   '{YL4_FIC4031.PV}/{YL4_FIC1840.PV}',
   '{YL4_FIC2075.PV}/{YL4_FIC1840.PV}',
   '{YL4_FIC1603.PV}/{YL4_FIC1840.PV}',
   '{YL4_FIC2117.PV}/{YL4_FIC1840.PV}',
   '{YL4_FIC2612.PV}/{YL4_FIC1840.PV}',
   '{YL4_FIC2230.PV}/{YL4_FIC1840.PV}',
   '{YL4_FIC2148-2.PV}/{YL4_FIC1840.PV}',
   '{YL4_FI2057_CALC.PV}/{YL4_FIC1840.PV}',
   '{YL4_FIC1827.PV}/{YL4_FIC1840.PV}',
   '{YL4_FIC1830_2.PV}/{YL4_FIC1840.PV}',
   '{YL4_FIC1830_1.PV}/{YL4_FIC1840.PV}',
   '{YL4_FIC1604.PV}/{YL4_FIC1840.PV}',
   '(CAPP("hf", "pt", UOMCONV({YL4_PI8640.PV}, "bar_g", "psia"), UOMCONV({YL4_TI8630A.PV}, "C", "F"), ARRAY(1), ARRAY("WA"), 19)-CAPP("hf", "pv", UOMCONV({YL4_PIC8665-1.PV}, "bar_a", "psia"), 0.9, ARRAY(1), ARRAY("WA"), 19))',
   #Raw Tags
   '{YL4_AI2113.PV}',
   '{YL4_AIC2108.PV}',
   '{YL4_AIT4050.PV}',
   '{YL4_E213_DT_CALC.PV}',
   '{YL4_TI1826.PV}',
   '{YL4_TI2017.PV}',
   '{YL4_TI2130.PV}',
   '{YL4_TI2610.PV}',
   '{YL4_TI2613.PV}',
   '{YL4_TIC2022.PV}',
   '{YL4_TIC3925.PV}'   
  ]

c = PLSModel()
c.load(filepath = '../Steam consumption_YL4_Residual_CJ_5_test.csv',
       y = 'Y1 [Steam consumption_residual (Steam consumption_residual)]',
       date = 'Time',Filter='YL4_FI9902')

c.Calculate(Filter=120)

c.Report()

c.Figure()

e = [0.000376583146987933,+0.000248329595077686,-1.26349145105701,+0.135617594868253,-0.0556814226377669,+0.83855233829073,+15.4040493652112,+0.0450554642316662,+0.0714875026968596,+0.0300832420594068,-0.390189990656601,-0.00863175529390242,-0.025636220318808,-0.0256130445744119,-0.0255903277575476,+5.02451221304183,-0.0322525173819052,+0.170067072156498,+0.220641345633089,-0.106963249716864,+5.09465125678271,+0.169269573125924,-0.475830873156258,+0.311327865859941,-0.243383214524775,+0.173609285129924,+0.186664263696956,-0.637045843971558,266.791547820943]

d = c.Formula.copy().T
d.columns=['Python']
d = d.assign(ProcessMD = e)
print(d)

#c.Optimize(item='SPEX',threshold=10)
#c.Optimize(item='SPEY',threshold=20)
#c.Report()
#c.Figure()

