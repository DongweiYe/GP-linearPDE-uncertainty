import numpy as np
import matplotlib.pyplot as plt

def visual_preprocess(x,y,xexact,yexact,xvague,yvague,show=False,save=False):
    plt.plot(x,y,'k-',linewidth=3,label='groundtruth')
    plt.plot(xexact,yexact,'r*',markersize=10,label='exact data')
    plt.plot(xvague,yvague,'b*',markersize=10,label='vague data ground truth')
    # plt.plot(X,10*vague_X_distribution+yvague_gt,'g-',label='vague distribution')
    plt.legend()
    if show == True:
        plt.show()
    if save == True:
        plt.savefig('preprocess.png',bbox_inches='tight')


def visual_prediction(x,y,xexact,yexact,xvague,yvague,ypred,show=False,save=False):
    plt.plot(x,y,'k-',linewidth=3,label='groundtruth')
    plt.plot(xexact,yexact,'r*',markersize=10,label='exact data')
    plt.plot(xvague,yvague,'b*',markersize=10,label='vague data ground truth')
    # plt.plot(X,10*vague_X_distribution+yvague_gt,'g-',label='vague distribution')
    plt.plot(x,ypred,linewidth=3,color='tab:purple',label='GP prediction (exact)')
    plt.legend()
    if show == True:
        plt.show()
    if save == True:
        plt.savefig('prediction.png',bbox_inches='tight')

# def plot_distribution():
