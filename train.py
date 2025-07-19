import scipy.io as sio
from sklearn.cluster import KMeans
from scipy.optimize import fmin_l_bfgs_b
from evaluation_metrics import *
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd

lambda1 = 1e-6
lambda2 = 1e-1
m = 12


def read_mat(url):
    data = sio.loadmat(url)
    return data


def compute_squared_EDM_method (D):
    m,n = D.shape
    G = np.dot(D, D .T)
    H = np.tile(np.diag(G), (m, 1))
    return H + H.T - 2*G

def compute_matrix_C(D):
    edm=compute_squared_EDM_method(D)
    c_temp=np.exp(-1/2*edm)
    row_sum=c_temp.sum(axis=1).reshape(c_temp.sum(axis=1).shape[0],1)
    C_norm=c_temp/row_sum
    return C_norm


def predict_func(x, m_theta, f_dim, l_dim):
    m_theta = m_theta.reshape(f_dim, l_dim)
    numerator = np.exp(np.dot(x, m_theta))
    denominator = np.sum(np.exp(np.dot(x, m_theta)), 1).reshape(-1, 1) + 0.00001
    return numerator / denominator

def obj_func1(w, x, d_, f_dim, l_dim):   #w, features, label_real, z, features_dim, labels_dim  
    w = w.reshape(f_dim, l_dim)   #(243,6)
    term1 = 0
    term2 = np.linalg.norm(w, ord=2)** 2
    term3 = 0.
    for i in range(m):
        P=predict_func(x[i], w, f_dim, l_dim)  #predict_result
        d_[i]=np.array(d_[i])

        term1+=np.sum(d_[i]*np.log((d_[i]+10**-6)/(P+10**-6)))

        C=compute_matrix_C(P)
        ln_C=np.log(C)
        C_entropy=np.multiply(C,ln_C)
        term3+=np.sum(C_entropy)
    loss1 = term1 + lambda1 * term2 + lambda2 * term3
    return loss1


def compute_PminusP_l(target_cow,l):
    m=target_cow.shape[0]
    target_cow_temp=target_cow.reshape(m,1)  #s×1
    target_cow=np.tile(target_cow,(m,1))     #1×s->s×s
    target_cow_temp=np.tile(target_cow_temp,(1,m))
    PminusP_l=target_cow-target_cow_temp
    return PminusP_l

def compute_gradient_PminusP_l_k(x,l,k,target_cow):
    x=np.array(x)
    gradient_P_l_k=(target_cow-target_cow*target_cow)*x[:,k] #(1×s)
    gradient_P_l_k_temp=gradient_P_l_k.reshape(gradient_P_l_k.shape[0],1)  #s×1
    lenth_g=len(gradient_P_l_k)  #s
    gradient_P_l_k=np.tile(gradient_P_l_k,(lenth_g,1))  #1×s->s×s
    gradient_P_l_k_temp=np.tile(gradient_P_l_k_temp,(1,lenth_g))   #s×1->s×s
    gradient_PminusP_l_k=gradient_P_l_k_temp-gradient_P_l_k
    return gradient_PminusP_l_k

def gradient_cij(w,x,features_dim,labels_dim):
    modProb = np.exp(np.dot(x, w))
    sumProb = np.sum(modProb, 1)
    disProb = modProb / (sumProb.reshape(-1, 1) + 0.000001)
    C=compute_matrix_C(disProb)
    gradient=np.zeros((features_dim,labels_dim))
    C_lnC=np.log(C)+np.ones_like(C)         
    for l in range(labels_dim):  #6
        for k in range(features_dim):   #243
            target_cow=disProb[:,l]
            PminusP=compute_PminusP_l(target_cow,l)
            gradient_PminusP=compute_gradient_PminusP_l_k(x,l,k,target_cow)

            matrix_temp=PminusP*gradient_PminusP*C

            sum_matrix=matrix_temp.sum(axis=1)
            m=sum_matrix.shape[0]
            sum_matrix = np.tile(sum_matrix, (m, 1))
            sum_matrix=sum_matrix.T

            result=matrix_temp-C*sum_matrix

            gradient[k][l]=np.sum(C_lnC*result)
    return gradient


def gradient_theta(w, x, d_,f_dim, l_dim):
    w = w.reshape(f_dim, l_dim)
    gradient = np.zeros_like(w)
    for i in range(m):
        modProb = np.exp(np.dot(x[i], w))
        sumProb = np.sum(modProb, 1)
        disProb = modProb / (sumProb.reshape(-1, 1) + 0.000001) 
        gradient+=np.transpose(x[i]).dot(disProb-d_[i])
        gradient +=lambda2 *gradient_cij(w,x[i],features_dim,labels_dim)
    gradient += 2 * lambda1 * w
    return gradient.ravel()


if __name__ == "__main__":
    data1 = read_mat(r"LDL-MES\datasets\Yeast_spoem.mat")   #data1:dict

    features = data1["features"]   #data1.keys()：labels,features    
    label_real = data1["labels"]
    features_dim = len(features[0])  #243
    labels_dim = len(label_real[0])  #6
    f = open("Yeast_spoem_m="+str(m)+".txt", "w")
    R=[[] for _ in range(11)]

    lambda11=[1e-5,1e-6,1e-7]
    lambda22=[1e-1,1e-2,1e-3,1e-4,1e-5]
    for lambda1 in lambda11:
        for lambda2 in lambda22:
            result1 = []
            result2 = []
            result3 = []
            result4 = []
            result5 = []
            result6 = []
            result7=[]
            result8 = []
            result9 = []
            result10 = []
            result11 =[]
            for t in range(3):
                x_train, x_test, y_train, y_test = train_test_split(features, label_real, test_size=0.2, random_state=2)
                # initialize
                w = np.random.rand(features_dim, labels_dim)    # update   (243,6)
                kmeans = KMeans(n_clusters=m).fit(y_train)
                kmeans_result = kmeans.predict(y_train)    #(170,)

                x_result = []
                d_result = []
                for i in range(m):
                    x_result.append([])
                    d_result.append([])
                for i in range(len(x_train)):   #
                    x_result[kmeans_result[i]].append(list(features[i]))
                    d_result[kmeans_result[i]].append(list(label_real[i]))

                # update step
                loss1 = obj_func1(w, x_result, d_result, features_dim, labels_dim)
                print(loss1)
                result = fmin_l_bfgs_b(obj_func1, w, gradient_theta, args=(x_result, d_result,features_dim, labels_dim),
                                        pgtol=1e-4, maxiter=100)
                w = result[0]
                pre_test = predict_func(x_test, w, features_dim, labels_dim)
                
                for i in range(y_test.shape[0]):
                    for j in range(y_test.shape[1]):
                        if y_test[i][j]==0:
                            y_test[i][j]=0.001
                # add each result to a list
                

                result7.append(euclidean(y_test, pre_test))
                print("No." + str(t) + ": " + str(euclidean(y_test, pre_test)))
                result8.append(sorensen(y_test, pre_test))
                print("No." + str(t) + ": " + str(sorensen(y_test, pre_test)))
                result9.append(squared_chi2(y_test, pre_test))
                print("No." + str(t) + ": " + str(squared_chi2(y_test, pre_test)))
                result10.append(fidelity(y_test, pre_test))
                print("No." + str(t) + ": " + str(fidelity(y_test, pre_test)))
                result11.append(squared_chord(y_test, pre_test))
                print("No." + str(t) + ": " + str(squared_chord(y_test, pre_test)))

                result1.append(chebyshev(y_test, pre_test))
                print("No." + str(t) + ": " + str(chebyshev(y_test, pre_test)))
                result2.append(clark(y_test, pre_test))
                print("No." + str(t) + ": " + str(clark(y_test, pre_test)))
                result3.append(canberra(y_test, pre_test))
                print("No." + str(t) + ": " + str(canberra(y_test, pre_test)))
                result4.append(kl(y_test, pre_test))
                print("No." + str(t) + ": " + str(kl(y_test, pre_test)))
                result5.append(cosine(y_test, pre_test))
                print("No." + str(t) + ": " + str(cosine(y_test, pre_test)))
                result6.append(intersection(y_test, pre_test))
                print("No." + str(t) + ": " + str(intersection(y_test, pre_test)))

            print(result1)
            print(result2)
            print(result3)
            print(result4)
            print(result5)
            print(result6)
            print(result7)
            print(result8)
            print(result9)
            print(result10)
            print(result11)

            print("euclidean:", np.mean(result7), "+", np.std(result7))
            print("sorensen:", np.mean(result8), "+", np.std(result8))
            print("squared_chi2:", np.mean(result9), "+", np.std(result9))
            print("fidelity:", np.mean(result10), "+", np.std(result10))
            print("squared_chord:", np.mean(result11), "+", np.std(result11))
            print("chebyshev:", np.mean(result1), "+", np.std(result1))
            print("clark:", np.mean(result2), "+", np.std(result2))
            print("canberra:", np.mean(result3), "+", np.std(result3))
            print("kl:", np.mean(result4), "+", np.std(result4))
            print("cosine:", np.mean(result5), "+", np.std(result5))
            print("intersection:", np.mean(result6), "+", np.std(result6))

            R[0].append(np.mean(result1))
            R[1].append(np.mean(result2))
            R[2].append(np.mean(result3))
            R[3].append(np.mean(result4))
            R[4].append(np.mean(result5))
            R[5].append(np.mean(result6))
            R[6].append(np.mean(result7))
            R[7].append(np.mean(result8))
            R[8].append(np.mean(result9))
            R[9].append(np.mean(result10))
            R[10].append(np.mean(result11))


            f.write("lambda1="+str(lambda1)+'\n'+"lambda2="+str(lambda2)+'\n'+"m="+str(m)+'\n')
            f.write("euclidean:"+str(np.mean(result7))+ "+"+str(np.std(result7))+'\n')
            f.write("sorensen:"+str(np.mean(result8))+ "+"+str(np.std(result8))+'\n')
            f.write("squared_chi2:"+str(np.mean(result9))+ "+"+str(np.std(result9))+'\n')
            f.write("fidelity:"+str(np.mean(result10))+ "+"+str(np.std(result10))+'\n')
            f.write("squared_chord:"+str(np.mean(result11))+ "+"+str(np.std(result11))+'\n')
            f.write("chebyshev:"+str(np.mean(result1))+ "+"+str(np.std(result1))+'\n')
            f.write("clark:"+str(np.mean(result2))+ "+"+str(np.std(result2))+'\n')
            f.write("canberra:"+str(np.mean(result3))+ "+"+str(np.std(result3))+'\n')
            f.write("kl:"+str(np.mean(result4))+ "+"+str(np.std(result4))+'\n')
            f.write("cosine:"+str(np.mean(result5))+ "+"+str(np.std(result5))+'\n')
            f.write("intersection:"+str(np.mean(result6))+ "+"+str(np.std(result6))+'\n'+'\n')

    name_list=['chebyshev','clark','canberra','kl','cosine','intersection','euclidean','sorensen','squared_chi2','fidelity','squared_chord']

    lenth1=len(lambda11)
    lenth2=len(lambda22)

    path="Yeast_spoem_m="+str(m)+".xlsx"
    excel_writer=pd.ExcelWriter(path,engine='openpyxl')
    
    for i in range(11):
        Ree=np.array(R[i]).reshape((lenth1,lenth2))
        data=pd.DataFrame(Ree,index=[['1e-5','1e-6','1e-7']],columns=[['1e-1','1e-2','1e-3','1e-4','1e-5']])
        data.to_excel(excel_writer,sheet_name=name_list[i])
    f.close()
    excel_writer.close()






















