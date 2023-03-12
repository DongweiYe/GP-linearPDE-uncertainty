import numpy as np

### Design the tuple data formation for Metro-Hasting algorithms
def bind_data(prior_mean,prior_variance,xexact,yexact,yvague,kernel):
    binded_data = (prior_mean,prior_variance,xexact,yexact,yvague,kernel)
    return binded_data



### This is also now for univariate
def prior_function(input_vector,mean_vector,variance_vector):
    return 1/np.sqrt((2*np.pi)*variance_vector)* \
            np.exp(-0.5*(input_vector-np.squeeze(mean_vector))**2/np.squeeze(variance_vector))


### databinding includes : 1. Prior information -> mean and variance
###                        2. Exact information -> xexact, yexact, yvague (belong to vague points but reflect the functions)
###                        3. GP information -> kernel
def Metropolis_Hasting(timestep,initial_sample,assumption_variance,databinding):
    ### Release databinding
    Xvague_prior_mean = databinding[0]
    Xvague_prior_var = databinding[1]
    Xexact = databinding[2]
    yexact = databinding[3]
    yvague = databinding[4]
    kernel = databinding[5]
    num_exact = yexact.shape[0]
    num_vague = yvague.shape[0]
    
    ### Initialise MCMC
    xvague_sample_current = initial_sample             ### Initial samples for each datapoints
    xvague_sample_list = np.empty((0,num_vague))       ### List of samples
    xvague_sample_list = np.vstack((xvague_sample_list,xvague_sample_current))

    ### MCMC sampling
    for t in range(timestep):

        ### Important! The workflow below this is now univaraite!!!
        x_new = np.abs(np.random.normal(np.squeeze(xvague_sample_current),assumption_variance,1))

        ### Component to compute multivariate Gaussian function for prior
        prior_function_upper = prior_function(x_new,Xvague_prior_mean,Xvague_prior_var)
        prior_function_lower = prior_function(xvague_sample_current,Xvague_prior_mean,Xvague_prior_var)
        
        ### Component to compute multivariate Gaussian function for likelihood, note here it is also noice free
        x_upper = np.append(Xexact,x_new)
        x_lower = np.append(Xexact,xvague_sample_current)

        y_vector = np.append(yexact,yvague)

        covariance_upper = kernel.K(np.expand_dims(x_upper,axis=1))
        covariance_lower = kernel.K(np.expand_dims(x_lower,axis=1))

        determinant_upper = np.linalg.det(covariance_upper)
        determinant_lower = np.linalg.det(covariance_lower)

        likelihood_upper = 1/np.sqrt((2*np.pi)**(num_exact+num_vague)*determinant_upper)*\
                                np.exp(-0.5*np.expand_dims(y_vector,axis=0)@np.linalg.inv(covariance_upper)@np.expand_dims(y_vector,axis=1))
        likelihood_lower = 1/np.sqrt((2*np.pi)**(num_exact+num_vague)*determinant_lower)*\
                                np.exp(-0.5*np.expand_dims(y_vector,axis=0)@np.linalg.inv(covariance_lower)@np.expand_dims(y_vector,axis=1))

        upper_alpha = prior_function_upper*likelihood_upper
        lower_alpha = prior_function_lower*likelihood_lower

        accept_ratio = np.squeeze(upper_alpha/lower_alpha) 
        check_sample = np.squeeze(np.random.uniform(0,1,1))

        if check_sample <= accept_ratio:
            xvague_sample_current = x_new
            xvague_sample_list = np.vstack((xvague_sample_list,xvague_sample_current))
            # print('Accept ratio: ',accept_ratio,'; Xnew: ',x_new,'; Accept')
        else:
            pass
            # print('Accept ratio: ',accept_ratio,'; Xnew: ',x_new,'; Reject')
    ### Truncate 1/4 of burning-in period sample
    truncate_num = int(xvague_sample_list.shape[0]/4)
    print('Number of posterior samples: ',xvague_sample_list[truncate_num:,:].shape[0])
    return xvague_sample_list[truncate_num:,:]
      