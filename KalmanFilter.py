from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import random

#Part B
def zeroNoise(K):
	#Where K is Kalman gain
	#Initial value of x0 = 100
	x = [100]
	itr = 0

	while itr < 100:
		x.append(x[itr]*(1-K))
		itr += 1
	return x

#Part C
def randomNoise(K, r):
	#Where K is Kalman Gain
	#Initial value of x0 = 100
	x = [100]
	itr = 0

	while itr < 100:
		#noise in Measurement
		wk = random.uniform(-r,r)
		#Noise of State
		vk = random.uniform(-r,r)
		x.append(x[itr]*(1-K) - K*wk + vk)
		itr += 1
	return x

#Part D
def KF(K, r):
	#Kalman Filter formula

	#Iterator
	itr = 0
	
	#Coefficients from equation xk+1 = Ak*xk + Bk*uk + vk
	Ak = 1
	Bk = 1
	#Coefficient from zk = Dk*xk
	Dk = 1

	#State coordinates xk:
	#xk|k
	xkk = [100]
	#xk+1|k
	xk1k = 0

	#Covariance Matrix Pk|k
	Pkk = [(r**2)/3]

	#Cov noise matricies
	Qk = (r**2)/3
	Rk = (r**2)/3

	while itr < 100:

		#Initialize Noise
		wk = random.uniform(-r,r)
		vk = random.uniform(-r,r)

		#Covariance Estimation
		#Prori State Cov Pk+1|k
		Pk1k = Ak*Pkk[itr]*Ak + Qk
		#Prior Measurement Cov Sk+1
		Sk1 = Dk*Pk1k*Dk + Rk
		#Kalman Gain
		Wk1 = Pk1k*Dk*(1/Sk1)
		Pkk.append(0)
		#Posteriori state cov Pk+1|k+1
		Pkk[itr+1] = Pk1k - Wk1*Sk1*Wk1


		#State Estimation
		#Priori state est xk+1|k
		xk1k = Ak*xkk[itr] + Bk*(-K*xkk[itr]) + vk
		#Priori measurement zk+1|k
		zk1k = Dk*xk1k
		#zk+1 
		zk1 = Dk*xkk[itr] + wk 
		xkk.append(0)
		#Posteriori state estimation
		xkk[itr+1] = xk1k + Wk1*(zk1 - zk1k)

		itr += 1
	
	#return xk|k
	return xkk

def SD(b, x):
	sum = 0
	itr = 0
	while itr < 100:
		sum += (b[itr]-x[itr])**2
		itr += 1
	return np.sqrt(sum/100)

if __name__ == '__main__':

	#Standard Deviation Analysis
	#No noise vs Random Noise r = 1 vs KM r = 1
	sd1r = SD(zeroNoise(0.5), randomNoise(0.5, 1))
	sd1k = SD(zeroNoise(0.5), KF(0.5, 1))
	print 'No noise vs Random noise and No noise vs Kalman Filter at r = 1'
	print (sd1r, sd1k)

	#No noise vs Random Noise r = 2.5 vs KM r = 2.5
	sd2r = SD(zeroNoise(0.5), randomNoise(0.5, 2.5))
	sd2k = SD(zeroNoise(0.5), KF(0.5, 2.5))
	print 'No noise vs Random noise and No noise vs Kalman Filter at r = 2.5'
	print (sd2r, sd2k)

	#No noise vs Random Noise r = 10 vs KM r = 10
	sd3r = SD(zeroNoise(0.5), randomNoise(0.5, 10))
	sd3k = SD(zeroNoise(0.5), KF(0.5, 10))
	print 'No noise vs Random noise and No noise vs Kalman Filter at r = 10'
	print (sd3r, sd3k)


	#Display graphs of data
	#Part B
	plt.plot(zeroNoise(0.5))
	plt.title("Zero Noise Case")
	plt.xlabel("Itr")
	plt.ylabel("X")
	plt.show()

	#Part C
	#r = 1
	plt.plot(randomNoise(0.5,1))
	plt.title("Random Noise with r = 1")
	plt.xlabel("Iteration")
	plt.ylabel("X")
	plt.show()

	#r = 2.5
	plt.plot(randomNoise(0.5,2.5))
	plt.title("Random Noise with r = 2.5")
	plt.xlabel("Iteration")
	plt.ylabel("X")
	plt.show()

	#r = 10
	plt.plot(randomNoise(0.5,10))
	plt.title("Random Noise with r = 10")
	plt.xlabel("Iteration")
	plt.ylabel("X")
	plt.show()

	#Part D
	#r = 1
	plt.plot(KF(0.5,1))
	plt.title("Kalman Filter with r = 1")
	plt.xlabel("Iteration")
	plt.ylabel("X")
	plt.show()

	#r = 2.5
	plt.plot(KF(0.5,2.5))
	plt.title("Kalman Filter with r = 2.5")
	plt.xlabel("Iteration")
	plt.ylabel("X")
	plt.show()

	#r = 10
	plt.plot(KF(0.5,10))
	plt.title("Kalman Filter with r = 10")
	plt.xlabel("Iteration")
	plt.ylabel("X")
	plt.show()
