import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm


def visual_data(x, y, xexact, yexact, xvague, xvague_prior_mean, yvague, show=False, save=False):
    plt.figure(figsize=(15, 3))
    plt.plot(x, y, 'k-', linewidth=3, label='groundtruth')
    plt.plot(xexact, yexact, 'ro', markersize=10, label='fixed data')
    plt.plot(xvague, yvague, 'bo', markersize=10, label='uncertain data ground truth')
    plt.plot(xvague_prior_mean, yvague, 'go', markersize=10, label='uncertain data prior mean')
    # plt.plot(X,10*vague_X_distribution+yvague_gt,'g-',label='vague distribution')

    for i in range(xvague.shape[0]):
        plt.arrow(xvague[i], yvague[i], xvague_prior_mean[i] - xvague[i], 0, color='grey')

    # plt.legend()
    if show == True:
        plt.show()
    if save == True:
        plt.savefig('preprocess.png', bbox_inches='tight')


def visual_GP(x, y, xexact, yexact, xvague, xvague_prior_mean, yvague, ypred, yvar, show=False, save=False):
    plt.figure(figsize=(15, 3))

    plt.plot(x, y, 'k-', linewidth=3, label='groundtruth')
    plt.plot(xexact, yexact, 'ro', markersize=10, label='fixed data')
    plt.plot(xvague, yvague, 'bo', markersize=10, label='uncertain data ground truth')
    plt.plot(xvague_prior_mean, yvague, 'go', markersize=10, label='uncertain data prior mean')

    plt.plot(x, ypred, linewidth=3, color='tab:purple', label='GP prediction direct')
    plt.fill_between(x, np.squeeze(ypred - np.sqrt(yvar)), np.squeeze(ypred + np.sqrt(yvar)), color='tab:purple',
                     alpha=0.3)
    for i in range(xvague.shape[0]):
        plt.arrow(xvague[i], yvague[i], xvague_prior_mean[i] - xvague[i], 0, color='grey')
    # plt.legend()
    if show == True:
        plt.show()
    if save == True:
        plt.savefig('GPprediction.png', bbox_inches='tight')


def vis_uncertain_input(x, y, xvague, xvague_prior_mean, xvague_post_mean, yvague, show=False, save=False):
    plt.plot(x, y, 'k-', linewidth=3, label='groundtruth')
    plt.plot(xvague, yvague, '*', color='black', markersize=10, label='uncertain data groundtruth')
    plt.plot(xvague_prior_mean, yvague, '*', color='tab:blue', markersize=10, label='uncertain prior')
    plt.plot(xvague_post_mean, yvague, '*', color='tab:red', markersize=10, label='uncertain posterior')

    # plt.plot(xvague_prior_mean,yvague,xvague_post_mean,yvague,color='tab:red')

    plt.legend()
    if show == True:
        plt.show()
    if save == True:
        plt.savefig('prediction.png', bbox_inches='tight')


def vis_prediction(x, y, \
                   xexact, yexact, \
                   xvague_prior, \
                   xvague_post, yvague, \
                   prior_ypred_list, prior_yvar_list, \
                   post_ypred_list, post_yvar_list, show=False, save=False):
    postymean_mean = np.mean(post_ypred_list, axis=0)
    postymean_var = np.var(post_ypred_list, axis=0)
    postyvar_mean = np.mean(post_yvar_list, axis=0)
    # postyvar_std = np.std(post_yvar_list,axis=0)

    post_lower_bound = np.squeeze(postymean_mean - 2 * np.sqrt(postymean_var + postyvar_mean))
    post_upper_bound = np.squeeze(postymean_mean + 2 * np.sqrt(postymean_var + postyvar_mean))

    priorymean_mean = np.mean(prior_ypred_list, axis=0)
    priorymean_var = np.var(prior_ypred_list, axis=0)
    prioryvar_mean = np.mean(prior_yvar_list, axis=0)
    # prioryvar_std = np.std(prior_yvar_list,axis=0)

    prior_lower_bound = np.squeeze(priorymean_mean - 2 * np.sqrt(priorymean_var + prioryvar_mean))
    prior_upper_bound = np.squeeze(priorymean_mean + 2 * np.sqrt(priorymean_var + prioryvar_mean))

    plt.figure(figsize=(17, 8.5))
    params = {
        'axes.labelsize': 25,
        'font.size': 25,
        'legend.fontsize': 25,
        'xtick.labelsize': 25,
        'ytick.labelsize': 25,
        'text.usetex': False,
        'axes.linewidth': 3,
        'xtick.major.width': 3,
        'ytick.major.width': 3,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
    }
    plt.rcParams.update(params)

    plt.fill_between(np.sort(np.append(x, xexact)), prior_lower_bound, prior_upper_bound, color='tab:orange', alpha=0.3)
    plt.fill_between(np.sort(np.append(x, xexact)), post_lower_bound, post_upper_bound, color='tab:purple', alpha=0.3)

    plt.plot(x, y, 'k--', linewidth=6, label='ground truth')
    plt.plot(np.sort(np.append(x, xexact)), priorymean_mean, '-', color='tab:orange', linewidth=6, label='with prior')
    plt.plot(np.sort(np.append(x, xexact)), postymean_mean, '-', color='tab:purple', linewidth=6,
             label='with posterior')

    plt.plot(xexact, yexact, 'rX', markersize=17, label='fixed data')
    plt.plot(xvague_prior, yvague, 'gX', markersize=17, label='uncertain data ground truth')
    plt.plot(xvague_post, yvague, 'bX', markersize=17, label='posterior mean of uncertain data')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0, 25])
    plt.ylim([-35, 25])

    plt.legend(loc='upper left', bbox_to_anchor=(0.0, -0.5), ncols=2, frameon=False)
    if show == True:
        plt.show()
    if save == True:
        plt.savefig('prediction.png', bbox_inches='tight')
    plt.savefig(f"prediction.pdf", format='pdf')


def visual_uncertainty(x, y, xexact, yexact, \
                       xvague, xvague_prior_mean, xvague_post_mean, yvague, \
                       preypred, preyvar, \
                       post_ypred_list, post_yvar_list, show=False, save=False):
    postymean_mean = np.mean(post_ypred_list, axis=0)
    postymean_std = np.std(post_ypred_list, axis=0)
    postyvar_mean = np.mean(post_yvar_list, axis=0)
    # postyvar_std = np.std(post_yvar_list,axis=0)

    post_lower_bound = np.squeeze(postymean_mean - postymean_std - postyvar_mean)
    post_upper_bound = np.squeeze(postymean_mean + postymean_std + postyvar_mean)

    ### Groundtruth
    plt.plot(x, y, 'k-', linewidth=3, label='groundtruth')
    plt.plot(xexact, yexact, '*', color='black', markersize=10, label='fixed data')

    ### Data plot
    # plt.plot(xvague,yvague,'*',color='grey',markersize=10,label='uncertain data groundtruth')
    # plt.plot(xvague_prior_mean,yvague,'*',color='tab:blue',markersize=10,label='uncertain prior')
    plt.plot(xvague_post_mean, yvague, '*', color='blue', markersize=10, label='uncertain posterior')

    ### Prior prediction
    # plt.plot(x,preypred,'r--',linewidth=3,label='GP wo vague')
    # plt.fill_between(x,np.squeeze(preypred-np.sqrt(preyvar)),np.squeeze(preypred+np.sqrt(preyvar)),color='red',alpha=0.3)

    ### Posterior prediction 
    plt.plot(x, postymean_mean, 'b--', linewidth=3, label='GP w vague inference')
    plt.fill_between(x, post_lower_bound, post_upper_bound, color='blue', alpha=0.3)

    # plt.plot(X,10*vague_X_distribution+yvague_gt,'g-',label='vague distribution')
    # plt.plot(x,ypred,linewidth=3,color='tab:purple',label='GP prediction (exact)')
    plt.legend()
    if show == True:
        plt.show()
    if save == True:
        plt.savefig('prediction.png', bbox_inches='tight')


def plot_distribution(prior_mean, prior_var, posterior_samples, groundtruth):
    if groundtruth.shape[0] == 1:
        ### Plot prior
        prior_range = np.arange(groundtruth - 4 * np.sqrt(prior_var), groundtruth + 4 * np.sqrt(prior_var),
                                8 * np.sqrt(prior_var) / 1000)
        plt.plot(prior_range, norm.pdf(prior_range, prior_mean, prior_var), 'b-', linewidth=3, label='Prior')
        ### Plot posterior
        posterior_range = np.arange(groundtruth - 4 * np.std(posterior_samples),
                                    groundtruth + 4 * np.std(posterior_samples), 8 * np.std(posterior_samples) / 1000)
        plt.plot(posterior_range, norm.pdf(posterior_range, np.mean(posterior_samples), np.var(posterior_samples)), '-',
                 color='tab:green', linewidth=3, label='Posterior')
        # plt.hist(posterior_samples, bins=10,alpha=0.3)

        ### Plot groundtruth
        plt.plot(groundtruth, 0, "r*", markersize=10, label='Groundtruth')
        # plt.xlim([np.min(prior_range),np.max(prior_range)])
    elif groundtruth.shape[0] == 2:
        ### Processing prior data
        sample_size = 10000
        prior_samples = np.random.multivariate_normal(prior_mean, np.diag(prior_var), sample_size)

        ### Plot prior
        sns.kdeplot(x=prior_samples[:, 0], y=prior_samples[:, 1], color='tab:blue')

        ### Plot posterior
        plt.plot(posterior_samples[:, 0], posterior_samples[:, 1], '.', color='tab:red', label='posterior samples',
                 alpha=0.1)
        sns.kdeplot(x=posterior_samples[:, 0], y=posterior_samples[:, 1], color='tab:green')

        ### Plot groundtruth
        plt.axvline(groundtruth[0], linestyle='--', color='black', label='groundtruth')
        plt.axhline(groundtruth[1], linestyle='--', color='black')

        ### 

    else:
        pass
    plt.legend()
    plt.show()


def figure1_standardGP(x, y, xexact, yexact, xvague, xvague_prior_mean, yvague, ypred, yvar, show=False, save=False):
    plt.figure(figsize=(17, 8.5))
    params = {
        'axes.labelsize': 25,
        'font.size': 25,
        'legend.fontsize': 25,
        'xtick.labelsize': 25,
        'ytick.labelsize': 25,
        'text.usetex': False,
        'axes.linewidth': 3,
        'xtick.major.width': 3,
        'ytick.major.width': 3,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
    }
    plt.rcParams.update(params)

    plt.fill_between(x, np.squeeze(ypred - 2 * np.sqrt(yvar)), np.squeeze(ypred + 2 * np.sqrt(yvar)),
                     color='tab:purple', alpha=0.3)

    plt.plot(x, y, 'k--', linewidth=6, label='ground truth')
    plt.plot(x, ypred, '-', linewidth=6, color='tab:purple', label='conventional GP')

    plt.plot(xexact, yexact, 'rX', markersize=20, label='fixed data')
    plt.plot(xvague, yvague, 'gX', markersize=20, label='uncertain data ground truth')
    plt.plot(xvague_prior_mean, yvague, 'bX', markersize=20, label='prior mean of uncertain data')

    # for i in range(xvague.shape[0]):
    #     plt.arrow(xvague[i],yvague[i],xvague_prior_mean[i]-xvague[i],0,color='grey')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0, 25])
    plt.ylim([-35, 25])
    plt.legend(loc='upper left', bbox_to_anchor=(0.0, -0.5), ncols=2, frameon=False)
    plt.savefig('Figure1_GP.png', bbox_inches='tight')


def figure1_posterior(prior_mean, prior_var, posterior_samples, groundtruth):
    ### for loopof number of uncertain data
    for i in range(prior_mean.shape[0]):
        plt.figure(figsize=(8, 8))
        params = {
            'axes.labelsize': 25,
            'font.size': 25,
            'legend.fontsize': 25,
            'xtick.labelsize': 25,
            'ytick.labelsize': 25,
            'text.usetex': False,
            'axes.linewidth': 3,
            'xtick.major.width': 3,
            'ytick.major.width': 3,
            'xtick.major.size': 5,
            'ytick.major.size': 5,
        }
        plt.rcParams.update(params)

        ### Plot prior
        prior_range = np.arange(prior_mean[i] - 6 * np.sqrt(prior_var[i]), prior_mean[i] + 6 * np.sqrt(prior_var[i]),
                                8 * np.sqrt(prior_var[i]) / 1000)
        plt.plot(prior_range, norm.pdf(prior_range, prior_mean[i], prior_var[i]), '-', color='royalblue', linewidth=6,
                 label='prior')
        # plt.axvline(x=prior_mean[i],color='tab:green',linestyle='--',linewidth=3)

        ### Plot groundtruth
        plt.axvline(x=groundtruth[i], color='black', linestyle='--', linewidth=6, label='ground truth')
        plt.xlim([np.min(prior_range), np.max(prior_range)])

        ### Plot posterior
        s1 = sns.kdeplot(posterior_samples[:, i], color='seagreen', bw_adjust=3, linewidth=6, label='posterior')
        plt.axvline(x=np.mean(posterior_samples[:, i]), color='seagreen', linestyle='--', linewidth=6,
                    label='posterior mean')
        s1.set(ylabel='')
        plt.xlabel('x')
        if i == 0:
            plt.legend(loc='upper left', bbox_to_anchor=(0.0, -0.5), ncols=4, frameon=False)
        plt.savefig('Figure1_Post_' + str(i) + '.png', bbox_inches='tight')


def figure2(x, y, \
            xexact, yexact, \
            xvague_gt, \
            xvague_prior, \
            xvague_post, yvague, \
            prior_ypred_list, prior_yvar_list, \
            post_ypred_list, post_yvar_list, show=False, save=False):
    postymean_mean = np.mean(post_ypred_list, axis=0)
    postymean_var = np.var(post_ypred_list, axis=0)
    postyvar_mean = np.mean(post_yvar_list, axis=0)
    # postyvar_std = np.std(post_yvar_list,axis=0)

    post_lower_bound = np.squeeze(postymean_mean - 2 * np.sqrt(postymean_var + postyvar_mean))
    post_upper_bound = np.squeeze(postymean_mean + 2 * np.sqrt(postymean_var + postyvar_mean))

    priorymean_mean = np.mean(prior_ypred_list, axis=0)
    priorymean_var = np.var(prior_ypred_list, axis=0)
    prioryvar_mean = np.mean(prior_yvar_list, axis=0)
    # prioryvar_std = np.std(prior_yvar_list,axis=0)

    prior_lower_bound = np.squeeze(priorymean_mean - 2 * np.sqrt(priorymean_var + prioryvar_mean))
    prior_upper_bound = np.squeeze(priorymean_mean + 2 * np.sqrt(priorymean_var + prioryvar_mean))

    plt.figure(figsize=(17, 8.5))
    params = {
        'axes.labelsize': 25,
        'font.size': 25,
        'legend.fontsize': 25,
        'xtick.labelsize': 25,
        'ytick.labelsize': 25,
        'text.usetex': False,
        'axes.linewidth': 3,
        'xtick.major.width': 3,
        'ytick.major.width': 3,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
    }
    plt.rcParams.update(params)

    # plt.plot(xexact,yexact,'rX',markersize=17,label='fixed data')

    plt.fill_between(x, prior_lower_bound, prior_upper_bound, color='tab:orange', alpha=0.3)
    plt.fill_between(x, post_lower_bound, post_upper_bound, color='tab:purple', alpha=0.3)

    plt.plot(x, y, 'k--', linewidth=7, label='ground truth')
    plt.plot(x, priorymean_mean, '-', color='tab:orange', linewidth=7, label='with prior', alpha=0.9)
    plt.plot(x, postymean_mean, '-', color='tab:purple', linewidth=7, label='with posterior', alpha=0.9)

    # plt.plot(xvague_gt,-15*np.ones(xvague_post.shape[0]),'gX',markersize=17,label='uncertain data ground truth')
    # plt.plot(xvague_post,-16.5*np.ones(xvague_post.shape[0]),'bX',markersize=17,label='posterior mean of uncertain data')
    # plt.plot(xvague_prior,-18*np.ones(xvague_post.shape[0]),'X',color='brown',markersize=17,label='prior mean of uncertain data')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0, 25])
    # plt.ylim([-21,17])

    plt.legend(loc='upper left', bbox_to_anchor=(0.0, -0.5), ncols=3, frameon=False)
    if show == True:
        plt.show()
    if save == True:
        plt.savefig('prediction.png', bbox_inches='tight')


def prediction_variance(ypred_list, yvar_list):
    ymean_var = np.var(ypred_list, axis=0)
    yvar_mean = np.mean(yvar_list, axis=0)

    return ymean_var + yvar_mean
