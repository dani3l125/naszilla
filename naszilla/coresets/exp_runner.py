import k_means_coreset_via_robust_median as pc
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("./HTRU_2.csv")
df=np.array(df)[:,:-1]
scaler = StandardScaler().fit(df)
df = scaler.transform(df)




if __name__ == '__main__':
    pc.run_multiple_exp(P=df,k=5,nqueries=1000,
                       centers=None,usr_ext='HTRU-small-sample',
                       repetitions = 10,mean_to_use=0,
                       std_to_use=1.5, use_threshold_method=True,
                       random_generation=False)

    pc.exp_plot(ext='HTRU-small-sample',repetitions=10,dir_to_save='figs')#use_threshold_method =  True)

