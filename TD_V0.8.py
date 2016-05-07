# coding: utf-8
# Code by SpinoPi
# Version:0.8

import pandas as pd
lType = ['AV01', 'AV02', 'AV03', 'AV04', 'AV05', 'AV06', 'AV07', 'AV08', 'AV09', 'AV10', 'AV11', 'AV12', 'AV13', 'AV14', 'AV15', 'AV16', 'AV17', 'AV18', 'AV19', 'AV20', 'AV21', 'AV22', 'AV23', 'AV24', 'AV25', 'AV26', 'AV27', 'AV28', 'AV29', 'AV30', 'AV31', 'AV32', 'CIS[1]', 'CIS[2]', 'COS[1]', 'COS[2]', 'CPV', 'CPV1', 'CPV2', 'CPV3', 'CSV', 'DV', 'ISS', 'MV', 'NALL', 'OSS', 'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'PV', 'PVP', 'RLV1', 'RMV', 'RSTS', 'RSV', 'RV', 'RV01', 'RV03', 'RV05', 'RV07', 'RV09', 'RV1', 'RV11', 'RV13', 'RV15', 'RV17', 'RV19', 'RV2', 'RV21', 'RV23', 'RV25', 'RV27', 'RV29', 'RV3', 'RV31', 'RV4', 'RV5', 'RV6', 'RV7', 'SSTS[1]', 'SSTS[2]', 'SUM', 'SV', 'TIME']

class tunning(object):
    def __init__(self,tag,type,value,lT):
        self._tag = tag
        self._type = type
        self._value = value.strip(';')
        self._paranum = value.count('=')
        self._parament = self.parament(lT)
        self.next = None

    def parament(self,lT):
        dp = {}
        for n in range(self._paranum):
            m = self._value.split(',')[n].split('=')
            dp[m[0]]=m[1]
            dp['TYPE']=self._type
            if len(lT)!=0:
                for n in lT:
                    if n in dp:
                        dp.pop(n)
            else:
                pass
        return dp

def Tunningdata(dA,lT):
    d = {}
    diMode = {'4194304':'AUT','2097152':'CAS','8388608':'MAN','142606336':'IMAN','75497472':'MAN-TRK','71303168':'AUT-TRK','138412032':'AUT-IMAN'}
    diAlarm = {'8388608':'NR','16384':'LO','278528':'LL','16416':'LO','27955':'LL','557184':'HH','557056':'HH','32768':'HH','1024':'DV-','2048':'DV+','16':'MLO','32':'MHI'}
    diAOF = {'1073741824':'AOF'}
    dA = dA.drop(len(dA)-1)
    for n in range(len(dA)-1):
        a = dA[0][n].split(':')
        b = tunning(a[-4],a[-2],a[-1],lT)
        d[b._tag] = b._parament
    dp = pd.DataFrame(d).T
    for n in ['MODE','OMOD','CMOD']:
        for m in diMode:
            dp[n] = dp[n].replace(m,diMode[m])
    for a in diAlarm:
        dp['ALRM'] = dp['ALRM'].replace(a,diAlarm[a])
    for f in diAOF:
        dp['AOFS'] = dp['AOFS'].replace(f,diAOF[f])
    return dp        

def index4match(dA,dB):
    l = []
    for n in range(len(dA)-1):
        if (dA.ix[n].name in dB.index)==True and (dA.ix[n].equals(dB.ix[dA.ix[n].name]))==False:
            l.append(dA.ix[n].name)
    return l

def compareTo(dA,dB):
    diA = {}
    for n in index4match(dA,dB):
        lt = dA.ix[n][pd.isnull(dA.ix[n]) == False]
        for m in lt.index:
            if (dA.ix[n][m] == dB.ix[n][m])==False:
                i = len(diA)
                diB = {}
                diB['Tag'] = n
                diB['Parament'] = m
                diB['Before'] = dA.ix[n][m]
                diB['After'] = dB.ix[n][m]
                diA[i] = diB
    return pd.DataFrame(diA,index=['Tag','Parament','Previous','New']).T

sA = input('The previous Tunningdata file :')
sB = input('The new Tunningdata file after:')
dA = pd.read_table(sA,skiprows=[0,1],header=None)
dB = pd.read_table(sB,skiprows=[0,1],header=None)
dC = compareTo(Tunningdata(dA,lType),Tunningdata(dB,lType))
dC.to_csv(sA.strip('.txt') + '_' + sB.strip('.txt') + '.csv')





