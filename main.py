import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t

#seaborn themes
sns.set_theme(
    style="whitegrid",      # clean background
    palette="deep",         # modern default colors
    font_scale=1.2          # slightly larger text
)



#actual linear regression
def train_regression(x,y,lr=0.01,epochs=100) -> tuple:
    #first set bias and weight to 0
    w=0.0
    b=0.0
    n=len(x)
    epochs=int(n/32)
    
    ws,bs,losses =[],[],[]
    for i in range(epochs):
        y_pred = b + w*x
        error = y-y_pred
        #calculate loss
        loss = np.mean(error**2)
        losses.append(loss)
        #update and change weight and bias
        ws.append(w)
        bs.append(b)
        dw = -2/n * np.sum(x*error)
        db = -2/n * np.sum(error)
        
        w-=lr*dw
        b-=lr*db
    #Slope = weight
    slope=w
    #Intercept = bias
    intercept = b
    #Correlation
    x_mean = x.mean()
    y_mean = y.mean()

    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    ss_yy = np.sum((y - y_mean) ** 2)

    correlation = ss_xy / np.sqrt(ss_xx * ss_yy)

    #Standard error not you normal one but slighlty changed for line of linear regression
    #formula is sqrt(s^2/mse)
    #s^2 = ssr/n-2
    #ssr (SUM OF SQUARED RESIDUALS) = sum((y-y_pred)**2)
    residuals = y-y_pred
    ssr = np.sum(residuals**2)
    s2=ssr/(n-2)
    
    std_err = np.sqrt(s2/ss_xx)
    
    #p-value calculation using scipy
    #NOTE: p-value is gives if the there is an actual relation between the two variables x and y
    #FORUMULA IS 
    t_stat = slope / std_err
    df = n - 2
    p_value = 2 * (1 - t.cdf(abs(t_stat), df=df))
    return w, b, correlation, p_value,std_err,losses,ws,bs 

c=input("Choose dataset 1 (success) or 2 (failure due to data):")
if c=="1":
    data = pd.read_csv('dataset_study.csv',usecols=[1,2])
    print(data)
    x=data["study_hours"]
    y=data['grade']
    x_name="study_hours"
    y_name='grade'
else:
    data = pd.read_csv('dataset.csv',usecols=[9,16],names=["Score","Hours"],skiprows=1)
    print(data)
    x=data["Hours"].astype('float64')
    y=data['Score'].astype('float64')
    x_name='Hours'
    y_name='Score'

#normalize data to try to imrpove performance and p-value    
x_mean = x.mean()
x_std = x.std()
x_norm = (x - x_mean) / x_std

y_mean = y.mean()
y_std = y.std()
y_norm = (y - y_mean) / y_std
mse_best=100000000000
best_lr=0
for lr in [0.1, 0.01, 0.001, 0.0001]:
    slope, intercept, correlation, p_value, std_err,losses_list,weights_list,biases_list = train_regression(x_norm, y_norm, lr=lr)
    y_pred= slope * x + intercept 
    mse = np.mean((y - y_pred) ** 2) 
    if mse<mse_best:
        mse_best = mse
        best_lr= lr
print("\033[92mBest lr=",best_lr)
slope, intercept, correlation, p_value, std_err,losses_list,weights_list,biases_list = train_regression(x_norm, y_norm,best_lr)

slope = slope * (y_std / x_std)
intercept = y_mean + y_std * intercept - slope * x_mean

x_line = np.linspace(min(x), max(x), 100)
y_line = slope * x_line + intercept

plt.figure()
plt.plot(losses_list)
plt.xlabel("Iteration")
plt.ylabel("MSE loss")
plt.title("Gradient descent: loss over iterations")
plt.show()

# 2) Weight and bias vs iteration
plt.figure()
plt.plot(weights_list, label="w (slope)")
plt.plot(biases_list, label="b (intercept)")
plt.xlabel("Iteration")
plt.ylabel("Parameter value")
plt.title("Gradient descent: parameters over iterations")
plt.legend()
plt.show()

sns.scatterplot(data=data,x=x_name,y=y_name,color="red")
plt.title("Linear Regression")
plt.plot(x_line,y_line,color="blue",label='slope')

y_pred= slope * x+ intercept 
mse = np.mean((y - y_pred) ** 2) 
rmse=np.sqrt(mse) 
print("\033[34mMSE=",mse)
print("\033[34mRMSE=",rmse)
if correlation < 0.70:
    print("\033[91mCorrelation=",correlation)
else:
    print("\033[92mCorrelation=",correlation)

alpha = 0.07
if p_value <= alpha:
    print("\033[92mSlope is statistically significant\033[0m")
else:
    print("\033[91mNo statistically significant linear relationship\033[0m")
print("\033[34p-value=",p_value)
print('\033[0m')
plt.show()