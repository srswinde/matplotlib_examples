from sklearn.svm import SVC
from sklearn.datasets.samples_generator import make_blobs
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic
from sklearn.gaussian_process import GaussianProcessClassifier
import numpy as np
import pylab




"""Using support vector machines to build a classification model which 
is then plotted using contourf. """

n_features = 2

def main():
	
	X, y = make_blobs(n_samples=1000, centers=2, n_features=n_features, random_state=0, cluster_std=0.5)
	fig = pylab.figure()
	fig.suptitle("SVM With RBF Kernel")
	pltnum = 1
	for gamma in range(1,10):
		ax = fig.add_subplot(4,4, pltnum )
		ax.set_title("gamma = {}".format(gamma))
		RBF(X,y, gamma, ax)
		pltnum+=1

	fig2 = pylab.figure()
	fig2.suptitle("SVM With Polynomial Kernel")
	pltnum=1

	for degree in range(1, 10):
		ax = fig2.add_subplot(3,3, pltnum )
		ax.set_title("degree = {}".format(degree))
		polynomial(X,y, degree, ax)
		pltnum+=1
			
        pylab.show()
	
		

def RBF( X,y, gamma, ax=None ):
	"""Function to model and plot my SVM with RBF kernal"""
	model = SVC(kernel="rbf", gamma=gamma, C=1.0 )
	model.fit(X,y)
	xx, yy = mesh(X)
	Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
	if ax is None:
		pylab.contourf(xx, yy, Z, cmap=pylab.cm.coolwarm, label="RBF" )
		pylab.plot(X[:,0][y==0], X[:,1][y==0], 'kx', label="Class 1")
		pylab.plot(X[:,0][y==1], X[:,1][y==1], 'k.', label="class 2")
	else:
		ax.contourf(xx, yy, Z, cmap=pylab.cm.coolwarm, label="RBF" )
		ax.plot(X[:,0][y==0], X[:,1][y==0], 'kx', label="Class 1")
		ax.plot(X[:,0][y==1], X[:,1][y==1], 'k.', label="class 2")
		
	pylab.legend()
	

def polynomial( X,y, degree=3, ax=None ):
	"""Function to model and then plot SVM with Polynomial kernal"""
	model =  SVC(kernel='poly', degree=degree, C=1.0)
	model.fit(X,y)
	xx,yy = mesh(X)
	Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
	if ax is None:
		pylab.contourf( xx, yy, Z, cmap=pylab.cm.coolwarm )
		pylab.plot( X[:,0][y==0], X[:,1][y==0], 'kx', label="Class 1" )
		pylab.plot( X[:,0][y==1], X[:,1][y==1], 'k.', label="class 2" )
	else:
		ax.contourf( xx, yy, Z, cmap=pylab.cm.coolwarm )
		ax.plot( X[:,0][y==0], X[:,1][y==0], 'kx', label="Class 1" )
		ax.plot( X[:,0][y==1], X[:,1][y==1], 'k.', label="class 2" )



X_train,Y_train = make_blobs(n_samples=1000, centers=2, n_features=n_features, random_state=0, cluster_std=0.5)

X_test, Y_test = make_blobs(n_samples=1000, centers=2, n_features=n_features, random_state=0, cluster_std=0.5)

def mesh(samples, step=0.2):
	Min = samples[:,0].min(), samples[:,1].min()
	Max = samples[:,0].max(), samples[:,1].max()
	xx, yy = np.meshgrid(np.arange(Min[0], Max[0], step ), np.arange(Min[1], Max[1], step ))
	return xx, yy
	

#rbf_clsf, rq_clsf = train( X_train, Y_train )

#test( rbf_clsf, rq_clsf, X_test, Y_test )

main()
