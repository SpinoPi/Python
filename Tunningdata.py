# coding: utf-8
# Code by SpinoPi
# Version:0.11
# This python program is aimming to compare two tunningdata file of YOKOGAWA

import pandas as pd
lType = ['AKLB[1]','AKLB[2]','AKLB[3]','AKLB[4]','AKLB[5]',
         'AV01', 'AV02', 'AV03','AV04', 'AV05', 'AV06', 'AV07', 'AV08', 'AV09', 'AV10',
         'AV11', 'AV12', 'AV13', 'AV14', 'AV15', 'AV16', 'AV17', 'AV18','AV19', 'AV20', 
         'AV21', 'AV22', 'AV23', 'AV24', 'AV25', 'AV26', 'AV27','AV28', 'AV29', 'AV30',
         'AV31', 'AV32', 
         'CIS[1]', 'CIS[2]', 'COS[1]', 'COS[2]', 
         'CPV', 'CPV1', 'CPV2', 'CPV3', 'CSV', 'DV', 'ISS', 'MV', 'NALL', 'OSS',
         'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08',
         'PT01','PT02','PT03','PT04', 'PT05','PT06','PT07','PT08','PT09','PT10',
         'PT11','PT12','PT13','PT14', 'PT15','PT16',
         'PV', 'PV01','PV02','PV03','PV04','PV05','PV06','PV07','PV08','PV09','PV10',
         'PV11','PV12','PV13','PV14','PV15','PV16',
         'PVP', 'RLV1','RMV', 'RSTS', 'RSV', 'RV',
         'RV01', 'RV03', 'RV05', 'RV07', 'RV09', 'RV1', 'RV11', 'RV13',
         'RV15', 'RV17', 'RV19', 'RV2', 'RV21', 'RV23', 'RV25', 'RV27',
         'RV29', 'RV3', 'RV31', 'RV4', 'RV5', 'RV6', 'RV7',
         'SSTS[1]', 'SSTS[2]', 'SUM', 'SV',
         'SWCR[1]','SWCR[2]','SWCR[3]','SWCR[4]','SWCR[5]',
         'SWLB[1]','SWLB[2]','SWLB[3]','SWLB[4]','SWLB[5]',
         'SWOP[1]','SWOP[2]','SWOP[3]','SWOP[4]','SWOP[5]',
         'SWST[1]','SWST[2]','SWST[3]','SWST[4]','SWST[5]',
         'TIME',]

class block(object):
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
        b = block(a[-4],a[-2],a[-1],lT)
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
                diB[dA.name] = dA.ix[n][m]
                diB[dB.name] = dB.ix[n][m]
                diA[i] = diB
    for n in set(dB.index).difference(set(dA.index)):
        i = len(diA)
        diB = {}
        diB['Tag'] = n
        diB['Parament'] = 'Block Change'
        diB[dB.name] = 'New'
        diA[i] = diB
    for n in set(dA.index).difference(set(dB.index)):
        i = len(diA)
        diB = {}
        diB['Tag'] = n
        diB['Parament'] = 'Block Change'
        diB[dB.name] = 'Deleted'
        diA[i] = diB
    return pd.DataFrame(diA,index=['Tag','Parament',dA.name,dB.name]).T

print('INSTRUCTION\nType "f" to get the paraments filted list.\nType "w" to get the sorted file from tunningdata.\nType "q" to quit.')
print('=====================================')
sA = input('The previous Tunningdata file :').strip('.txt')
if (sA == 'f' or sA == 'F'):
    print('These paraments had been filted:')
    print(lType.sort())
    sA = input('The previous Tunningdata file :').strip('.txt')
elif (sA == 'w' or sA =='W'):
    sC = input('The Tunningdata file :').strip('.txt')
    dC = pd.read_table(sC + '.txt',skiprows=[0,1],header=None)
    dD = Tunningdata(dC,lType)
    dD.to_csv(sC.strip('.txt') + '.csv')
    print('The result had been saved into' + sC.strip('.txt') + '.csv')
    print('Print any key to quit.')
    input()
    quit()
elif (sA == 'q' or sA == 'Q'):
    quit()
sB = input('The new Tunningdata file:').strip('.txt')
dA = pd.read_table(sA + '.txt',skiprows=[0,1],header=None)
dB = pd.read_table(sB + '.txt',skiprows=[0,1],header=None)
dC = Tunningdata(dA,lType)
dD = Tunningdata(dB,lType)
dC.name = sA
dD.name = sB
dE = compareTo(dC,dD)
sC = sA + '_' + sB + '.csv'
dE.to_csv(sC)
print('The result had been saved into ' + sC)
print('Print any key to quit.')
input()
