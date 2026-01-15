import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
import logging
import time
import sys
import datetime
import os

real_input = input

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
        self.is_newline = True  

    def write(self, message):
        self.terminal.write(message)
        
        if not self.log.closed:
            for char in message:
                if self.is_newline and char != '\n':
                    # add timestamp only at the start of a non-empty line
                    timestamp = datetime.datetime.now().strftime("[%H:%M:%S] ")
                    self.log.write(timestamp)
                    self.is_newline = False
                
                self.log.write(char)
                
                if char == '\n':
                    self.is_newline = True

    def flush(self):
        self.terminal.flush()
        if not self.log.closed:
            self.log.flush()

    def __del__(self):
        if hasattr(self, 'log') and not self.log.closed:
            self.log.close()



filename = os.path.splitext(os.path.basename(__file__))[0]
try:
    os.mkdir("logs")
except FileExistsError:
    pass
except OSError as e:
    print(f"An error occurred: {e}")

safe_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = os.path.join("logs", f"{filename}@{safe_timestamp}.log")
sys.stdout = Logger(log_path)

def input_and_log(prompt=""):
    print(prompt, end="", flush=True)
    answer = real_input()
    if not sys.stdout.log.closed:
        sys.stdout.log.write(answer + "\n")
        sys.stdout.log.flush()
    return answer

input = input_and_log

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



def choose_dataset() -> None:
    print("Choose dataset 1 (success) or 2 (failure due to data):")
    c = input() 
    
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
    return x,y,x_name,y_name,data

def result() -> None:
    #normalize data to try to imrpove performance and p-value   
    x,y,x_name,y_name,data = choose_dataset() 
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
    print("Best lr=",best_lr)
    slope, intercept, correlation, p_value, std_err,losses_list,weights_list,biases_list = train_regression(x_norm, y_norm,best_lr)


    slope = slope * (y_std / x_std)
    intercept = y_mean + y_std * intercept - slope * x_mean


    x_line = np.linspace(min(x), max(x), 100)
    y_line = slope * x_line + intercept

    layout = [
        ["loss", "params"],
        ["scatter", "scatter"]
    ]

    fig, axes = plt.subplot_mosaic(layout, figsize=(10, 8))

    # 1) Loss vs Iteration (Top Left)
    axes["loss"].plot(losses_list)
    axes["loss"].set_xlabel("Iteration")
    axes["loss"].set_ylabel("MSE loss")
    axes["loss"].set_title("Gradient descent: loss over iterations")

    # 2) Weights and Bias vs Iteration (Top Right)
    axes["params"].plot(weights_list, label="w (slope)")
    axes["params"].plot(biases_list, label="b (intercept)")
    axes["params"].set_xlabel("Iteration")
    axes["params"].set_ylabel("Parameter value")
    axes["params"].set_title("Parameters over iterations")
    axes["params"].legend()

    # 3) Scatter Plot (Bottom, Spanning Full Width)
    sns.scatterplot(data=data, x=x_name, y=y_name, color="red", ax=axes["scatter"])
    axes["scatter"].plot(x_line, y_line, color="blue", label='slope')
    axes["scatter"].set_title("Linear Regression")
    axes["scatter"].legend()

    # Adjust spacing to prevent overlap
    plt.tight_layout()
    plt.show()
        
    
    y_pred= slope * x+ intercept 
    mse = np.mean((y - y_pred) ** 2) 
    rmse=np.sqrt(mse) 
    print("MSE=",mse)
    print("RMSE=",rmse)
    if correlation < 0.70:
        print("Correlation=",correlation)
    else:
        print("Correlation=",correlation)


    alpha = 0.07
    if p_value <= alpha:
        print("Slope is statistically significant")
    else:
        print("No statistically significant linear relationship")
    print("p-value=",p_value)
    
    plt.show()
    
if __name__ =="__main__":
    result()
