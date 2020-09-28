# python imports
import os
import glob
import sys
import random

import numpy as np
import scipy.io as sio
import utils
import subspacemodels
import kernels
import dists



if __name__ == '__main__':

    #load data
    #training_data=numpy array of size M,N where M is the data dimension an N is the number of samples. In our MICCAI paper, we use the SCT/JSRT data set. Please see the code files related to our Medical Image Analysis paper (https://imi.uni-luebeck.de/multi-resolution-multi-object-statistical-shape-models) on how to obtain and convert that data.

    #test_data= see training data

    M,N=training_data.shape
    #objects indices for the SCR/JSRT data when pre-processed as described above
    obj_indicator=np.zeros(M)
    obj_indicator[0:88]=0 #right lung
    obj_indicator[88:188]=1 #left lung
    obj_indicator[188:240]=2 #heart
    obj_indicator[240:286]=3 #right clavicle
    obj_indicator[286:332]=4 #left clavicle
    color_map=['b','k','r','g','y']
    spacing=0.35  
    training_data=training_data*spacing
    test_data=test_data*spacing
    spec_samples=1000
    alpha_spec=5
    spec_mode='normalbouned'
        
    # Select a (small) training data set

    size_training_set=[5,10,15,20,30,40,70,120];
    size_test_set=test_data.shape[1]
    num_random_runs=[50,50,50,50,50,50,50,50]

    models=[]
    gen_results=[]
    spec_results=[]
    basis_sizes=[]

    num_levels=5

    #compute geodesic distance matrix only once on all training images to save time during training (of course, introduces small bias)

    #multi-object distance proposed in our MedIA paper   
 distance_matrix_sp=utils.compute_multi_object_pseudo_euclidean_geodesic_shortest_path_2d_point_distance_matrix(np.mean(training_data,axis=1,keepdims=True),obj_indicator,eta=10,kappa=150)
    max_distance=np.max(distance_matrix_sp)
    distance_schedule=max_distance*np.power(0.5,np.array(range(0,num_levels)))

    #euclidean distance
    distance_matrix_eucl=utils.compute_euclidean_2d_point_distance_matrix(np.mean(training_data,axis=1,keepdims=True))
    max_distance_eucl=np.max(distance_matrix_eucl)
    distance_schedule_eucl=max_distance_eucl*np.power(0.5,np.array(range(0,num_levels)))

    #run main loop
    for curr_index in range(0,len(size_training_set)):
        curr_size_training_set=size_training_set[curr_index]
        curr_num_random_runs=num_random_runs[curr_index]

        curr_models=[[],[],[],[]]
        curr_results=[[],[],[],[]]
        curr_spec=[[],[],[],[]]
        curr_size_basis=[[],[],[],[]]

        print('\nSize training set: '+str(curr_size_training_set)+' size test set: '+str(size_test_set)+' runs: '+str(curr_num_random_runs))

        #run each random experiment
        for run in range(0,curr_num_random_runs):

            #(randomly) select the data entries for training and test
            training_idx=np.random.choice(training_data.shape[1],curr_size_training_set,replace=False)
            sample_data_matrix=training_data[:,training_idx];
            sample_test_data=test_data;

            #generate standard pca model
            variability_retained=0.95

            pca_model=subspacemodels.SubspaceModelGenerator.compute_pca_subspace(np.matrix(sample_data_matrix),variability_retained)
            approx,error=pca_model.approximate(sample_test_data,utils.mean_error_2d_contour)
            spec=pca_model.compute_specificity(sample_test_data,alpha_spec, utils.mean_error_2d_contour, num_samples=spec_samples,mode=spec_mode)

            curr_spec[0].append(np.mean(spec))
            curr_models[0].append(pca_model)
            curr_results[0].append(np.mean(error))
            curr_size_basis[0].append(pca_model.basis.shape[1])
       
            #generate LSSM model
           localized_model=subspacemodels.SubspaceModelGenerator.compute_localized_subspace_media(np.matrix(sample_data_matrix),variability_retained,distance_matrix_sp,distance_schedule,sample_test_data,merge_method=utils.merge_subspace_models_closest_rotation_decorr)
            approx,error=localized_model.approximate(sample_test_data,utils.mean_error_2d_contour)
            spec=localized_model.compute_specificity(sample_test_data,alpha_spec, utils.mean_error_2d_contour, num_samples=spec_samples,mode=spec_mode)

            curr_spec[1].append(np.mean(spec))
            curr_models[1].append(localized_model)
            curr_results[1].append(np.mean(error))
            curr_size_basis[1].append(localized_model.basis.shape[1])        

            #generate KLSSM+Grass model
            contour_distance=dists.SimpleMatrixDist(distance_matrix_eucl)
            cov_kernel=kernels.CovKernel(1/(sample_data_matrix.shape[1]-1))
            gamma=1/(2*((2*distance_schedule_eucl)**2))
            exponent=2

            kernels_lvl_1=[(cov_kernel,None,'data',None,1)]
            kernels_lvl_2=[(cov_kernel,None,'data',None,1), (kernels.ExponentialKernel(gamma[1],exponent),np.multiply,'dist',contour_distance,1)]
            kernels_lvl_3=[(cov_kernel,None,'data',None,1), (kernels.ExponentialKernel(gamma[2],exponent),np.multiply,'dist',contour_distance,1)]
            kernels_lvl_4=[(cov_kernel,None,'data',None,1), (kernels.ExponentialKernel(gamma[3],exponent),np.multiply,'dist',contour_distance,1)]
            kernels_lvl_5=[(cov_kernel,None,'data',None,1), (kernels.ExponentialKernel(gamma[4],exponent),np.multiply,'dist',contour_distance,1)]

            kernel_list=[kernels_lvl_1,kernels_lvl_2,kernels_lvl_3,kernels_lvl_4,kernels_lvl_5]
            max_rank=331

            localized_model_kernel_eucl2=subspacemodels.SubspaceModelGenerator.compute_localized_subspace_kernel(np.matrix(sample_data_matrix),variability_retained,distance_schedule_eucl,kernel_list,max_rank,sample_test_data,debug=False,eig_method=lambda_eig_method,merge_method=utils.merge_subspace_models_closest_rotation_decorr_kernel)
            approx,error=localized_model_kernel_eucl2.approximate(sample_test_data,utils.mean_error_2d_contour)
            spec=localized_model_kernel_eucl2.compute_specificity(sample_test_data,alpha_spec, utils.mean_error_2d_contour, num_samples=spec_samples,mode=spec_mode)

            curr_spec[2].append(np.mean(spec))
            curr_models[2].append(localized_model_kernel_eucl2)
            curr_results[2].append(np.mean(error))
            curr_size_basis[2].append(localized_model_kernel_eucl2.basis.shape[1])

            #generate KLSSM+Kernel model        
            cov_kernel=kernels.CovKernel(1/(sample_data_matrix.shape[1]-1))

            kernel_list=[[(kernels.OnesKernel(),None,'dist',contour_distance,.2), (kernels.ExponentialKernel(gamma[1],exponent),np.add,'dist',contour_distance,.2),(kernels.ExponentialKernel(gamma[2],exponent),np.add,'dist',contour_distance,.2), (kernels.ExponentialKernel(gamma[3],exponent),np.add,'dist',contour_distance,.2), (kernels.ExponentialKernel(gamma[4],exponent),np.add,'dist',contour_distance,.2), (cov_kernel,np.multiply,'data',None,1)]]

            localized_model_kernel_gpmm_eucl2=subspacemodels.SubspaceModelGenerator.compute_localized_subspace_kernel(np.matrix(sample_data_matrix),variability_retained,[0],kernel_list,max_rank,sample_test_data,eig_method=lambda_eig_method)
            approx,error=localized_model_kernel_gpmm_eucl2.approximate(sample_test_data,utils.mean_error_2d_contour)
            spec=localized_model_kernel_gpmm_eucl2.compute_specificity(sample_test_data,alpha_spec, utils.mean_error_2d_contour, num_samples=spec_samples,mode=spec_mode)

            curr_spec[3].append(np.mean(spec))
            curr_models[3].append(localized_model_kernel_gpmm_eucl2)
            curr_results[3].append(np.mean(error))
            curr_size_basis[3].append(localized_model_kernel_gpmm_eucl2.basis.shape[1])

            ## end random runs

        models.append(curr_models)
        gen_results.append(curr_results)
        basis_sizes.append(curr_size_basis)
        spec_results.append(curr_spec)

        print('\nSSM model: '+str(np.mean(curr_results[0]))+' size: '+str(np.mean(curr_size_basis[0]))+' spec: '+str(np.mean(curr_spec[0])))
        print('\nLocalized models (Geodesic)')
        print('\nLSSM: '+str(np.mean(curr_results[1]))+' size: '+str(np.mean(curr_size_basis[1]))+' spec: '+str(np.mean(curr_spec[1])))
        print('\nKLSSM+Grass: '+str(np.mean(curr_results[2]))+' size: '+str(np.mean(curr_size_basis[2]))+' spec: '+str(np.mean(curr_spec[2])))
        print('\nKLSSM+Kernel: '+str(np.mean(curr_results[3]))+' size: '+str(np.mean(curr_size_basis[3]))+' spec: '+str(np.mean(curr_spec[3])))
        

    ##end training set size

    print('\nPress any key to continue')

    input()

    

    


    
    
