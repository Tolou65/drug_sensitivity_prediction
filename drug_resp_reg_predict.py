from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
#from rlscore.utilities. import KronRLS
from rlscore.measure import cindex
import numpy as np
from scipy.stats import pearsonr,spearmanr
from rlscore.measure import sqerror
import scipy.io
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform



"""importing data sets"""

#ge=np.genfromtxt('original_input/cell_lines_geneExpression.csv', delimiter=",", skip_header=1, usecols=np.arange(1,13322,1))
Y=np.genfromtxt('GDSC_drugResponse__drugs_VS_cellLines.csv', delimiter="\t")
#fp=np.genfromtxt('original_input/thesis_fp_pub.csv', delimiter=",", skip_header=1)


"""importing calculated kernels"""
#k_cell= np.loadtxt('CCA_triton_res_eval/KC_GK__geneExpr_sigma_147.txt')
for l in np.arange(1,6,1):
    k_drug=np.loadtxt("DrugKernel/prepro_std_85_1run_5fols_1stcom/stdfp_95_1stcomp_result_fold"+str(l)+".mat")


"""Random gene selection"""
#np.random.seed(seed=123)
#rand_col=np.sort(np.random.randint(13322, size=0))
#print(rand_col)
#ge_rand=ge[:,rand_col]
#print(ge.shape)


"""importing PCA for ge"""
#Z = scipy.io.loadmat('Z.mat')
#ge = np.array(Z['z'])

"""gene selection by variance"""
# gene_variance=ge.var(0)#print(rand_col)
#
# ge_var=ge[:,gene_variance>10]
# print(ge_var.shape)
# #print(ge.shape)

MSE_diff_pca_comp=np.zeros(9)
""" generating random matrix for using instead of gene expression"""
ge=np.random.uniform(low=4.43, high=14.88, size=(124,50))
#print(ge.shape)
s=0
for l in np.arange(1,6,1):
    filename = "CellKer_ind_limge5_3rep_5f/5com_MultiKer_Uni/geK_var5_uniMulti_kernel_rep" + str(i + 1) + ".mat"
    prepro_std_fp_mat= scipy.io.loadmat(filename)
    #limfp_mat= scipy.io.loadmat('CCA_triton_res_eval/pubfp_nonzero_result_run_num_5.mat')
    k_drug=np.array(limge_mat['FivComp_uniKer'])
        cell_tr=np.array(np.where(limge_mat['tr_ind']==1)[1])
        cell_ts=np.array(np.where(limge_mat['tr_ind']==0)[1])
    ###bring the projected ge and train and test indeces
    #ge_pca=ge[:,1:l]

    """ calculating the gaussian kernel with median trick"""
    pairwise_sq_dists = squareform(pdist(ge, 'sqeuclidean'))
    s=np.median(pairwise_sq_dists)
    K = np.exp(-pairwise_sq_dists**2 / s**2)


    #Y=np.genfromtxt('GDSC_drugResponse__drugs_VS_cellLines.csv', delimiter="\t")
    #print(Y.shape)
    #resp=scipy.io.loadmat('CellKer_ind_limge5_3rep_5f/5com_MultiKer_Uni/scaled_resp.mat')
    #Y=np.array(resp['y'])
    #k_cell= np.loadtxt('CCA_triton_res_eval/KC_GK__geneExpr_sigma_147.txt')
    #k_drug=np.loadtxt('CCA_triton_res_eval/KD_PubChem.txt')

    """calculating the kron product"""
    K=np.kron(k_drug,K)
    #DTI_vec= np.ravel(Y, 'F')
    reg_par=np.arange(-3,3,1)
    #N=len(DTI_vec)
    Y=np.ravel(Y,'F')

    mask=np.arange(0, len(Y), 1)

    cv_out = KFold(n_splits=10, shuffle=True, random_state=123)
    cv_in = KFold(n_splits=3, shuffle=True, random_state=123)
    mse_o=np.zeros(cv_out.get_n_splits())
    cind_o=np.zeros(cv_out.get_n_splits())
    pear_o=np.zeros(cv_out.get_n_splits())
    spear_o=np.zeros(cv_out.get_n_splits())
    k=0
    MSE=np.zeros(( cv_in.get_n_splits() , reg_par.size))
    for train_out , test_out in cv_out.split(mask):
        inner_mask=mask[train_out]
        i=0 #inner fold counter
        for train_in , test_in in cv_in.split(inner_mask):
            train_in_index=inner_mask[train_in]
            test_in_index=inner_mask[test_in]
            j=0 #reg param counter
            for alpha in reg_par:
                a=10.0**alpha
        # Y_train_in=Y(mask_train_in)
        # Y_test_in = Y(mask_test_in)

            K_train=K[train_in_index][:,train_in_index]
            K_test = K[test_in_index][:, train_in_index]
            Y_train=Y[train_in_index]
            Y_test = Y[test_in_index]
        #print(Y_train.shape)
        #print(Y_test.shape)
            clf = KernelRidge(alpha=a)
            clf.fit(K_train, Y_train)
            pred=clf.predict(K_test)
            #print(pred.shape)
            #print(Y_train.shape)
            mse=np.sqrt(sqerror(np.ravel(Y_test,'F'),np.ravel(pred,'F')))
            MSE[i,j]=mse
        MSE_m = np.mean(MSE, axis=0)
        opt_regpram = 10. ** (reg_par[np.argmin(MSE_m)])
        clf = KernelRidge(alpha=opt_regpram)
        K_train_o=K[train_out][:,train_out]
        K_test_o = K[test_out][:, train_out]
        Y_train_o = Y[train_out]
        Y_test_o = Y[test_out]
        clf.fit(K_train_o, Y_train_o)
        pred_o = clf.predict(K_test_o)
        mse_o[k] = np.sqrt(sqerror(np.ravel(Y_test_o, 'F'), np.ravel(pred_o, 'F')))
        cind_o[k] = cindex(np.ravel(Y_test_o, 'F'), np.ravel(pred_o, 'F'))
        pear_o[k] = pearsonr(np.ravel(Y_test_o, 'F'), np.ravel(pred_o, 'F'))[0]
        spear_o[k] = spearmanr(np.ravel(Y_test_o, 'F'), np.ravel(pred_o, 'F'))[0]
        #plt.scatter(np.ravel(Y_test_o, 'F'), np.ravel(pred_o, 'F'))
        #plt.show()
        print("op_reg:%f, fold RMSE:%f, fold Cindex:%f, fold Pearson:%f, fold Spearman:%f" % (opt_regpram,mse_o[k],
                                                                                         cind_o[k],
                                                                                         pear_o[k],
                                                                                         spear_o[k]))
        k+=1
    #MSE_diff_pca_comp[s]=np.mean(mse_o)
    #s+=1
    print("overall RMSE:%f, overall Cindex:%f, overall Pearson:%f, overall Spearman:%f" % (np.mean(mse_o),
                                                                                     np.mean(cind_o),
                                                                                     np.mean(pear_o),
                                                                                   np.mean(spear_o)))
    #MSE_diff_pca_comp[s] = np.mean(mse_o)
    #s+=1
    #print(pred.shape)
    #res=clf.score(K_test,Y_test)
    #print(res)
    # plt.plot(MSE_diff_pca_comp)
    # plt.show()
    # file = open("diff_pca_comp_results.txt","w")
    # file.write(MSE_diff_pca_comp)
    # file.close()
