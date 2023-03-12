import numpy as np


### Compute covariance matrix with input vector
def covariance_matrix(input_vector,lengthscale,magnitude):
    D = distance.squareform(distance.pdist(X, 'sqeuclidean'))

    lengthscale = np.exp(parameters[0])
    magnitude = np.exp(parameters[1])
    return magnitude**2 * np.exp(-D/lengthscale) 

def prior_function(input_vector,mean_vector,variance_vector):
    return 1/np.sqrt((2*np.pi)*assumption_variance)* \
            np.exp(-0.5*(x_new-np.squeeze(Xvague_prior_mean))**2/np.squeeze(Xvague_prior_var))



# def mh_process():
      