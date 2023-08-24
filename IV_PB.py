import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#Black-Schoels price szamolo fuggveny
def my_BS(S, K, r, T, sigma, flag):
    d1 = (np.log(S / K) + (r + 0.5 * sigma*sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if flag == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif flag == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price

#fuggveny, ami kiszamolja azt az implied volatilityt, amivel a Black-Schoels model eredménye kozel van a valosaghoz
def my_IV(S, K, r, T, sigma, flag, P_real, eps, correction):
	P_th=my_BS(S, K, r, T, sigma, flag)
	i=0
	while abs(P_real-P_th)>eps:
		if P_real-P_th>0:
			sigma+=correction
		else:
			sigma-=correction
		P_th=my_BS(S, K, r, T, sigma, flag)
		i+=1
		if i>1000:
			return sigma
	return sigma

#fuggveny, ami kiszamolja adott datumhoz es lejarathoz az IV feluletet
def my_surface_1(data,date1,date2,init_IV,flag): 
	D=data.loc[date1]
	D=D.loc[lambda D: D['cp_flag'] == flag]
	D=D.loc[lambda D: D['expiry'] == date2]
	D=D.sort_values('strike')
	T=np.datetime64(date2)-np.datetime64(date1)
	T = (T.astype(float))/365

	sh=D.shape
	n=sh[0]
	IV_arr=np.empty([n,2])
	for i in range(n):
		O=D.iloc[i,:]
		K=O.iloc[8]
		S=O.iloc[6]
		P_real=(O.iloc[4]+O.iloc[5])/2
		IV_arr[i,0]=my_IV(S, K, r, T, sigma_init, 'call', P_real, 1, 0.001)
		IV_arr[i,1]=K

	return IV_arr

####################################################################################################################
#fuggveny, ami minden megfigyeleshez kiszamolja a lejaratig hatralevo idot
def my_data_transform(D,flag): #flag C vagy P
	D = D[D['cp_flag'] == flag]
	init_date=data.index
	end_date=data.iloc[:,6]
	sh=D.shape
	n=sh[0]
	T=np.empty(n)
	for i in range(n):
		in_T=init_date[i]
		in_T=np.datetime64(in_T.strftime('%Y-%m-%d'))
		end_T=end_date[i].to_numpy()
		Ti=end_T-in_T
		Ti=Ti.astype('timedelta64[D]')
		T[i] = Ti.astype(int)
	D.insert(8, "rem_time", T)

	return D

#fuggveny ami letrehozza a 3D felulethez szukseges IV matrixot, amely tartalmaz minden lehetseges strike-lejarat kombinaciot
def my_surface_2(data,init_IV,r): 
	K_=np.sort(np.unique(data[:,7]))
	T_=np.sort(np.unique(data[:,8]))
	IV=np.empty([K_.size, T_.size])
	print(K_.size,'   ',T_.size)
	for k in range(K_.size):
		for t in range(T_.size):
	#for k in range(3):
		#for t in range(3):		
			D=data[data[:,7] == K_[k]]
			D=D[D[:,8] == T_[t]]
			if D.size==0:
				IV[k,t]=IV[k,t-1]
			else: 
				#if sh[0]>1:
				D=D[0,:]
				K=D[7]
				T=D[8]/365
				S=D[5]
				P_real=(D[3]+D[4])/2
				IV[k,t]=my_IV(S, K, r, T, init_IV, 'call', P_real, 0.5, 0.001)
			#print('t: ',t)
		print('k: ',k,'','k%: ',k/K_.size*100)
	return K_, T_, IV			


data=pd.read_parquet('SPX_opt_data.pq')

#datum alapjan IV felulet
r=0.001
sigma_init=0.1
date1='2015-01-02'
date2='2015-03-20'
iv=my_surface_1(data,date1,date2,sigma_init,'C')
figure,axes=plt.subplots()
axes.plot(iv[:,1],iv[:,0])
plt.show()

#adat transzformacio 
data2=my_data_transform(data,'C')
D_np=data2.to_numpy()
path="C:/d"
np.save(path+'data2_np',D_np,allow_pickle=True)
D_np=np.load(path+'data2_np.npy',allow_pickle=True)

#3D plothoz input szamolas
ms=my_surface_2(D_np,sigma_init,r)
np.save(path+'ms0',ms[0],allow_pickle=True)
np.save(path+'ms1',ms[1],allow_pickle=True)
np.save(path+'ms2',ms[2],allow_pickle=True)

print(ms[0])
print(ms[1])
print(ms[2])
