
# coding: utf-8

from math import *

def MerkelNumber(t_inl,t_out,ts,P,R):

    def Ps(t):
        tj = t + 273.15
        x=33.590624-3142.305/tj-8.2*(log(tj)/log(10))+0.0024804*tj
        return (10**x)/1000
    
    def K(t):
        #...
        return 1-t/(586-0.56*(t-20))
    
    def h_w(t):
        #...
        return (1555 + 1.14448 * t) * Ps(t) / (P - Ps(t))
    
    def h(t):
        #...
        return h_w(ts) + 4.1868 * (t - t_out) / (K(t_out) * R)

    def func(t): 
        #...
        return 1/(h_w(t)-h(t))

    def Get_N(a,b,width):
        N=int((b-a)/width + 1)
        if N%2 == 0:
            N=N+1
        return N

    def GenerateData(a,b,n,width):
        datas = []
        r=a
        
        for i in range(0,n):
            datas.append(func(r))
            r = r+width
        return datas

    def simpson_integral(datas,width,n):
        sum = datas[0]+datas[n-1]
        for i in range(2,n):
            if i%2== 0:
                sum = sum +4*datas[i-1]
            else:
                sum = sum +2*datas[i-1]
        return sum*width/3.0
    
    a=t_out
    b=t_inl
    width=2**-10
    N=Get_N(a,b,width)
    datas = GenerateData(a,b,N,width)
    return simpson_integral(datas,width,N)*4.1868/K(t_out)
