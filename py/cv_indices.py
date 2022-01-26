import numpy as np
import argparse
 
### This script writes index files for cross validation for easy transfer to matlab

# X/Y : nx3x3
def get_kfold_indices(n, k, savepath=None):

    arr=np.zeros((k,n), dtype=int)
    n_i = int(n/k)

    for i in np.arange(k):
        if i == k-1:
            arr[i, i*n_i:]=1
        else:
            arr[i, i*n_i:(i+1)*n_i]=1

    if savepath is not None:
        np.savetxt(savepath+"/cv_idx.csv", arr, delimiter=',', fmt='%i')
    print("CV indices for %i samples and %i folds are:\n" % (n,k))
    print(arr)

    return arr

 


parser = argparse.ArgumentParser()
parser.add_argument("idx_path", help="idx_path.", type=str)
parser.add_argument("n", help="n", type=str)
parser.add_argument("k", help="k", type=str)

args = parser.parse_args()
args.k=int(args.k)
args.n=int(args.n)


get_kfold_indices(args.n, args.k, args.idx_path)








#def get_fold_i(X,idx,i):
#    return X[np.where(idx[i,:]==1)]

#      acc_score = []
#     model.fit(X_train,y_train)
#     pred_values = model.predict(X_test)
     
#     acc = accuracy_score(pred_values , y_test)
#     acc_score.append(acc)
     
# avg_acc_score = sum(acc_score)/k
 
# print('accuracy of each fold - {}'.format(acc_score))
# print('Avg accuracy : {}'.format(avg_acc_score))