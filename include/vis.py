import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

def visual_data(x,y,xexact,yexact,xvague,xvague_prior_mean,yvague,show=False,save=False):
    plt.figure(figsize=(15, 3))
    plt.plot(x,y,'k-',linewidth=3,label='groundtruth')
    plt.plot(xexact,yexact,'ro',markersize=10,label='fixed data')
    plt.plot(xvague,yvague,'bo',markersize=10,label='uncertain data ground truth')
    plt.plot(xvague_prior_mean,yvague,'go',markersize=10,label='uncertain data prior mean')
    # plt.plot(X,10*vague_X_distribution+yvague_gt,'g-',label='vague distribution')
    
    for i in range(xvague.shape[0]):
        plt.arrow(xvague[i],yvague[i],xvague_prior_mean[i]-xvague[i],0,color='grey')
    
    # plt.legend()
    if show == True:
        plt.show()
    if save == True:
        plt.savefig('preprocess.png',bbox_inches='tight')


def visual_GP(x,y,xexact,yexact,xvague,xvague_prior_mean,yvague,ypred,yvar,show=False,save=False):
    plt.figure(figsize=(15, 3))

    plt.plot(x,y,'k-',linewidth=3,label='groundtruth')
    plt.plot(xexact,yexact,'ro',markersize=10,label='fixed data')
    plt.plot(xvague,yvague,'bo',markersize=10,label='uncertain data ground truth')
    plt.plot(xvague_prior_mean,yvague,'go',markersize=10,label='uncertain data prior mean')
    
    plt.plot(x,ypred,linewidth=3,color='tab:purple',label='GP prediction direct')
    plt.fill_between(x,np.squeeze(ypred-np.sqrt(yvar)),np.squeeze(ypred+np.sqrt(yvar)),color='tab:purple',alpha=0.3)
    for i in range(xvague.shape[0]):
        plt.arrow(xvague[i],yvague[i],xvague_prior_mean[i]-xvague[i],0,color='grey')
    # plt.legend()
    if show == True:
        plt.show()
    if save == True:
        plt.savefig('GPprediction.png',bbox_inches='tight')


def vis_uncertain_input(x,y,xvague,xvague_prior_mean,xvague_post_mean,yvague,show=False,save=False):
    
    plt.plot(x,y,'k-',linewidth=3,label='groundtruth')
    plt.plot(xvague,yvague,'*',color='black',markersize=10,label='uncertain data groundtruth')
    plt.plot(xvague_prior_mean,yvague,'*',color='tab:blue',markersize=10,label='uncertain prior')
    plt.plot(xvague_post_mean,yvague,'*',color='tab:red',markersize=10,label='uncertain posterior')
    
    # plt.plot(xvague_prior_mean,yvague,xvague_post_mean,yvague,color='tab:red')
    
    plt.legend()
    if show == True:
        plt.show()
    if save == True:
        plt.savefig('prediction.png',bbox_inches='tight')

def vis_prediction(x,y,\
                   prior_ypred_list,prior_yvar_list,\
                    post_ypred_list,post_yvar_list,show=False,save=False):
    postymean_mean = np.mean(post_ypred_list,axis=0)
    postymean_var = np.var(post_ypred_list,axis=0)
    postyvar_mean = np.mean(post_yvar_list,axis=0)
    # postyvar_std = np.std(post_yvar_list,axis=0)

    post_lower_bound = np.squeeze(postymean_mean-np.sqrt(postymean_var+postyvar_mean))
    post_upper_bound = np.squeeze(postymean_mean+np.sqrt(postymean_var+postyvar_mean))
    
    priorymean_mean = np.mean(prior_ypred_list,axis=0)
    priorymean_var = np.var(prior_ypred_list,axis=0)
    prioryvar_mean = np.mean(prior_yvar_list,axis=0)
    # prioryvar_std = np.std(prior_yvar_list,axis=0)

    prior_lower_bound = np.squeeze(priorymean_mean-np.sqrt(priorymean_var+prioryvar_mean))
    prior_upper_bound = np.squeeze(priorymean_mean+np.sqrt(priorymean_var+prioryvar_mean))

    plt.figure(figsize=(15, 3))
    plt.plot(x,y,'k-',linewidth=3,label='groundtruth')

    plt.plot(x,priorymean_mean,'--',color='tab:blue',linewidth=3,label='with prior')
    plt.fill_between(x,prior_lower_bound,prior_upper_bound,color='tab:blue',alpha=0.3)

    plt.plot(x,postymean_mean,'--',color='tab:red',linewidth=3,label='with posterior')
    plt.fill_between(x,post_lower_bound,post_upper_bound,color='tab:red',alpha=0.3)

    

    plt.legend()
    if show == True:
        plt.show()
    if save == True:
        plt.savefig('prediction.png',bbox_inches='tight')


def visual_uncertainty(x,y,xexact,yexact,\
                        xvague,xvague_prior_mean,xvague_post_mean,yvague,\
                        preypred,preyvar,\
                        post_ypred_list,post_yvar_list,show=False,save=False):
    
    postymean_mean = np.mean(post_ypred_list,axis=0)
    postymean_std = np.std(post_ypred_list,axis=0)
    postyvar_mean = np.mean(post_yvar_list,axis=0)
    # postyvar_std = np.std(post_yvar_list,axis=0)

    post_lower_bound = np.squeeze(postymean_mean-postymean_std-postyvar_mean)
    post_upper_bound = np.squeeze(postymean_mean+postymean_std+postyvar_mean)
    
    ### Groundtruth
    plt.plot(x,y,'k-',linewidth=3,label='groundtruth')
    plt.plot(xexact,yexact,'*',color='black',markersize=10,label='fixed data')

    ### Data plot
    # plt.plot(xvague,yvague,'*',color='grey',markersize=10,label='uncertain data groundtruth')
    # plt.plot(xvague_prior_mean,yvague,'*',color='tab:blue',markersize=10,label='uncertain prior')
    plt.plot(xvague_post_mean,yvague,'*',color='blue',markersize=10,label='uncertain posterior')

    ### Prior prediction
    # plt.plot(x,preypred,'r--',linewidth=3,label='GP wo vague')
    # plt.fill_between(x,np.squeeze(preypred-np.sqrt(preyvar)),np.squeeze(preypred+np.sqrt(preyvar)),color='red',alpha=0.3)

    ### Posterior prediction 
    plt.plot(x,postymean_mean,'b--',linewidth=3,label='GP w vague inference')
    plt.fill_between(x,post_lower_bound,post_upper_bound,color='blue',alpha=0.3)
    
    

    # plt.plot(X,10*vague_X_distribution+yvague_gt,'g-',label='vague distribution')
    # plt.plot(x,ypred,linewidth=3,color='tab:purple',label='GP prediction (exact)')
    plt.legend()
    if show == True:
        plt.show()
    if save == True:
        plt.savefig('prediction.png',bbox_inches='tight')

def plot_distribution(prior_mean,prior_var,posterior_samples,groundtruth):
    if groundtruth.shape[0] == 1:
        ### Plot prior
        prior_range = np.arange(groundtruth-4*np.sqrt(prior_var), groundtruth+4*np.sqrt(prior_var), 8*np.sqrt(prior_var)/1000)
        plt.plot(prior_range, norm.pdf(prior_range, prior_mean, prior_var),'b-',linewidth=3,label='Prior')
        ### Plot posterior
        posterior_range = np.arange(groundtruth-4*np.std(posterior_samples), groundtruth+4*np.std(posterior_samples), 8*np.std(posterior_samples)/1000)
        plt.plot(posterior_range, norm.pdf(posterior_range, np.mean(posterior_samples), np.var(posterior_samples)),'-',color='tab:green',linewidth=3,label='Posterior')
        # plt.hist(posterior_samples, bins=10,alpha=0.3)

        ### Plot groundtruth
        plt.plot(groundtruth, 0,"r*",markersize=10,label='Groundtruth')
        # plt.xlim([np.min(prior_range),np.max(prior_range)])
    elif groundtruth.shape[0] == 2:
        ### Processing prior data
        sample_size = 10000
        prior_samples = np.random.multivariate_normal(prior_mean,np.diag(prior_var),sample_size)
        
        ### Plot prior
        sns.kdeplot(x=prior_samples[:,0], y=prior_samples[:,1], color='tab:blue') 

        ### Plot posterior
        plt.plot(posterior_samples[:,0],posterior_samples[:,1],'.',color='tab:red',label='posterior samples',alpha=0.1)
        sns.kdeplot(x=posterior_samples[:,0], y=posterior_samples[:,1], color='tab:green')

        ### Plot groundtruth
        plt.axvline(groundtruth[0], linestyle='--',color='black',label='groundtruth')
        plt.axhline(groundtruth[1], linestyle='--',color='black')
    
        ### 
    
    else:
        pass
    plt.legend()
    plt.show()
