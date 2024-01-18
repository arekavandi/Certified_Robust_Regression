''' This code was developed by Aref Miri Rekavandi @2024 based on DSAC* repository, for the manuscript entitled 
"RS-Reg: Probabilistic and Robust Certified Regression Against Adversarial Attack Through Randomized Smoothing".
If you used our code in your research, please cite the aforementioned work.
'''
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn.functional as F
import numpy as np
import cv2

import dsacstar

import time
import argparse
import math

from dataset import CamLocDataset
from network import Network


import random
#import numpy as np
from scipy.stats import norm
from scipy.stats import binom
from scipy.optimize import brentq
import matplotlib.pyplot as plt

def lowerboundestim(p):
        return 1-binom.cdf(accepted, n_tr, p)-alpha/2
    
def difference_between_cdfs(p):
    if ((U-value-epsilon)/(U-value-tau))>((value-epsilon-L)/(value-tau-L)):
        return binom.cdf(np.ceil(n_sample*((epsilon-tau)/(U-value-tau))), n_sample, 1-p)-P
    else:
        return binom.cdf(np.ceil(n_sample*((epsilon-tau)/(value-tau-L))), n_sample, 1-p)-P  
    
def difference_between_cdfs_dis(p):
    upper=1-np.min([beta*np.abs(epsilon)/(U-value-epsilon),1])
    lower=1-np.min([beta*np.abs(epsilon)/(value-epsilon-L),1])
    if upper>lower:
        print('Upper is active, value:',value)
       # print(np.min([beta*np.abs(epsilon)/(U-value-epsilon),1]))
        return binom.cdf(np.ceil(n_sample*((np.min([beta*np.abs(epsilon)/(U-value-epsilon),1])))), n_sample, 1-p)-P
    else:
        print('Lower is active, value:',value)
       # print(np.min([beta*np.abs(epsilon)/(value-epsilon-L),1]))
        return binom.cdf(np.ceil(n_sample*((np.min([beta*np.abs(epsilon)/(value-epsilon-L),1])))), n_sample, 1-p)-P

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(
        description='Test a trained network on a specific scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('scene', help='name of a scene in the dataset folder, e.g. Cambridge_GreatCourt')

    parser.add_argument('network', help='file name of a network trained for the scene')

    parser.add_argument('--hypotheses', '-hyps', type=int, default=64, 
        help='number of hypotheses, i.e. number of RANSAC iterations')
    
    parser.add_argument('--n_tr', '-ntr', type=int, default=200, 
        help='number of training samples to identify the network')
    
    parser.add_argument('--n_sample', '-ns', type=int, default=10, 
        help='number of samples to compute the smoothed value')
                        
    parser.add_argument('--alpha', '-alpha', type=float, default=0.5, 
        help='confidence level between 0 to 1')
    
    parser.add_argument('--P', '-P', type=float, default=0.8, 
        help='user-defined probability of observing valid output')

    parser.add_argument('--K', '-K', type=float, default=1.5, 
        help='Penalty term in certified error')

    parser.add_argument('--U', '-U', type=float, default=85, 
        help='upper bound on the output value')

    parser.add_argument('--L', '-L', type=float, default=-15, 
        help='Lower bound on the output value')

    parser.add_argument('--epsilon', '-ep', type=float, default=5, 
        help='validity range of output')

    parser.add_argument('--tau', '-tau', type=float, default=0.5, 
        help='asymptotic range of expected output value')

    parser.add_argument('--beta', '-beta', type=float, default=2, 
        help='discount factor') 

    parser.add_argument('--n_test_image', '-nt', type=int, default=120, 
        help='number of evaluated test images') 

    parser.add_argument('--threshold', '-t', type=float, default=10, 
        help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')

    parser.add_argument('--inlieralpha', '-ia', type=float, default=100, 
        help='alpha parameter of the soft inlier count; controls the softness of the hypotheses score distribution; lower means softer')

    parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100, 
        help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking pose consistency towards all measurements; error is clamped to this value for stability')

    parser.add_argument('--mode', '-m', type=int, default=1, choices=[1,2],
        help='test mode: 1 = RGB, 2 = RGB-D')

    parser.add_argument('--tiny', '-tiny', action='store_true',
        help='Load a model with massively reduced capacity for a low memory footprint.')

    parser.add_argument('--session', '-sid', default='',
        help='custom session name appended to output files, useful to separate different runs of a script')

    opt = parser.parse_args()

    # setup dataset
    if opt.mode < 2: opt.mode = 0 # we do not load ground truth scene coordinates when testing
    testset = CamLocDataset("D:/dsacstar-master/datasets/" + opt.scene + "/test", mode = opt.mode)
    testset_loader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=6)

    # load network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = Network(torch.zeros((3)), opt.tiny)
    network.load_state_dict(torch.load(opt.network, map_location=torch.device('cpu')))
  #  network.load_state_dict(torch.load(opt.network))
    network = network.to(device)
    network.eval()

    print('Test images found: ', len(testset))

    ### Initialization of other parameters#################################
    sigma=[0.025, 0.05];   r = np.linspace(0, 0.1, 15)
    n_tr=opt.n_tr; n_sample=opt.n_sample; alpha=opt.alpha
    P=opt.P; K=opt.K; U=opt.U; L=opt.L
    epsilon=opt.epsilon; tau=opt.tau; beta=opt.beta
    n_test_image=opt.n_test_image
    ######################################################################
    print('Numebr of evaluated test images: ', n_test_image)
    
    with torch.no_grad():	
        for index in range(len(sigma)):
            Cer_error=[]
            Cer_error_g=[]
            Cer_error_gd=[]
            ex=np.zeros(n_test_image)
            ex_g=np.zeros(n_test_image)
            ex_gd=np.zeros(n_test_image)
            for radius in r:
                tErrs = []   
                tgErrs=[]
                tgdErrs=[]
                image_ind=0
                for image, gt_pose, init, focal_length, file in testset_loader:
                    if image_ind==n_test_image:
                        break

                    focal_length = float(focal_length[0])
                    gt_pose = gt_pose[0]
                    image = image.to(device)
                    print('Image size: ',image.shape)

                    if radius==0:
                        scene_coordinates = network(image)
                        scene_coordinates = scene_coordinates.cpu()

                        ref_pose = torch.zeros((4, 4))

                        dsacstar.forward_rgb(
                                scene_coordinates, 
                                ref_pose, 
                                opt.hypotheses, 
                                opt.threshold,
                                focal_length, 
                                float(image.size(3) / 2), #principal point assumed in image center
                                float(image.size(2) / 2), 
                                opt.inlieralpha,
                                opt.maxpixelerror,
                                network.OUTPUT_SUBSAMPLE)            
                        accepted=0
                        counter=0
                        print('-------------------------------------------------------')
                        print(f'Reference Coordinate:{ref_pose[0:3,3]}m')
                        print('-------------------------------------------------------')
                        g=torch.zeros((4, 4))
                        while counter<n_tr:
                            # predict scene coordinates and neural guidance
                            noise=torch.randn(image.shape)* torch.tensor(sigma[index])
                            scene_coordinates = network(image+noise)
                            scene_coordinates = scene_coordinates.cpu()

                            out_pose = torch.zeros((4, 4))

                            dsacstar.forward_rgb(
                                scene_coordinates, 
                                out_pose, 
                                opt.hypotheses, 
                                opt.threshold,
                                focal_length, 
                                float(image.size(3) / 2), #principal point assumed in image center
                                float(image.size(2) / 2), 
                                opt.inlieralpha,
                                opt.maxpixelerror,
                                network.OUTPUT_SUBSAMPLE)
                            if torch.max(torch.abs(ref_pose[0:3, 3] - out_pose[0:3, 3]))<epsilon:
                                accepted+=1
                            counter+=1
                            
                        if accepted==counter:
                            accepted-=1
                        elif accepted==0:
                            accepted=1
                        probability_lower = brentq(lowerboundestim, 0, 1)
                        ex[image_ind]=sigma[index]*(norm.ppf(probability_lower, 0, 1)-norm.ppf(P, 0, 1))
                        temp=np.zeros(3)
                        for out_index in range(3):
                            value=out_pose[out_index, 3]
                            temp[out_index]=brentq(difference_between_cdfs, 0, 1)
                        probability_success = np.min(temp)
                        print('probability_success is:',probability_success)
                        ex_g[image_ind]=sigma[index]*(norm.ppf(probability_lower, 0, 1)-norm.ppf(probability_success, 0, 1))
                        
                        temp=np.zeros(3)
                        for out_index in range(3):
                            value=out_pose[out_index, 3]
                            temp[out_index]=brentq(difference_between_cdfs_dis, 0, 1)
                        probability_dis = np.min(temp)
                        print('probability_dis is:',probability_dis)
                        ex_gd[image_ind]=sigma[index]*(norm.ppf(probability_lower, 0, 1)-norm.ppf(probability_dis, 0, 1))
                         
                    counter=0    
                    adv=torch.randn(image.shape)* torch.tensor(sigma[index])
                    adv=adv/torch.norm(adv, p=2)
                    adv=torch.tensor(random.random()*radius)*adv
                    g=torch.zeros((4, 4))
                    while counter<n_sample:
                        noise=torch.randn(image.shape)* torch.tensor(sigma[index])
                        
                        #plotimage=image+adv+noise
                        #plt.figure()
                        #imgplot = plt.imshow(plotimage[0,0,:,:],cmap='gray')
                        #plt.colorbar()
                        #plt.title(f'Attacked Image with sigma={sigma[index]}, r<{radius}')
                        #plt.show()
                        
                        scene_coordinates = network(image+adv+noise)
                        scene_coordinates = scene_coordinates.cpu()

                        out_pose = torch.zeros((4, 4))

                        dsacstar.forward_rgb(
                                scene_coordinates, 
                                out_pose, 
                                opt.hypotheses, 
                                opt.threshold,
                                focal_length, 
                                float(image.size(3) / 2), #principal point assumed in image center
                                float(image.size(2) / 2), 
                                opt.inlieralpha,
                                opt.maxpixelerror,
                                network.OUTPUT_SUBSAMPLE)
                        g+=out_pose/n_sample
                        counter+=1
                        
                    t_err = float(torch.norm(gt_pose[0:3, 3] - out_pose[0:3, 3]))
                    t_err_g = float(torch.norm(gt_pose[0:3, 3] - g[0:3, 3]))
                    t_err_gd = float(torch.norm(gt_pose[0:3, 3] - g[0:3, 3]))
                    if radius>ex[image_ind]:
                        t_err+=K
                    if radius>ex_g[image_ind]:
                        t_err_g+=K
                    if radius>ex_gd[image_ind]:
                        t_err_gd+=K
                    
                    print('--------------------------------------------------------------------------------------------------------')
                    print('Which image?:', image_ind)
                    print('Which radius?:', radius)
                    print('Input L2-bound for f(x) is:', ex[image_ind])
                    print('Input L2-bound for g(x) is:', ex_g[image_ind])
                    print('Input L2-bound for discounted g(x) is:', ex_gd[image_ind])
                    print('Certified Error f(x), g(x), and discounted g(x):', t_err * 100, t_err_g * 100, t_err_gd * 100)
                    print('--------------------------------------------------------------------------------------------------------')
                    image_ind+=1
                    tErrs.append(t_err * 100)
                    tgErrs.append(t_err_g * 100)
                    tgdErrs.append(t_err_gd * 100)

                median_idx = int(len(tErrs)/2)
                tErrs.sort()
                tgErrs.sort()
                tgdErrs.sort()
                print("\nMedian Error for f(x): %.1fcm" % (tErrs[median_idx]))
                print("\nMedian Error for g(x): %.1fcm" % (tgErrs[median_idx]))
                print("\nMedian Error for discounted g(x): %.1fcm" % (tgdErrs[median_idx]))
                Cer_error.append(tErrs[median_idx])
                Cer_error_g.append(tgErrs[median_idx])
                Cer_error_gd.append(tgdErrs[median_idx])
            plt.plot(r,Cer_error,label=f"f(x), sigma={sigma[index]}")
            plt.plot(r,Cer_error_g,label=f"g(x), sigma={sigma[index]}")
            plt.plot(r,Cer_error_gd,label=f"Dis. g(x), sigma={sigma[index]}")

        print("\n===================================================")
        print("\nTest complete.")
    
    plt.grid(True)
    plt.xlabel('radius')
    plt.ylabel('certified error')
    plt.legend(loc='upper left') 
    plt.show()
