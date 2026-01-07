import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
import logging
import time
import sys
import datetime

#configure logging to save to file .log
logging.basicConfig(
    filename=f'Linear_regression_run_@_{datetime.datetime.now()}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    filemode='w'
)
logger = logging.getLogger(__name__)

#class to redirect print statements to logger with line buffering
class StreamToLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.buffer = "" #add buffer to hold partial prints

    def write(self, message):
        #append new message to buffer
        self.buffer += message
        
        #check if buffer contains newline
        while "\n" in self.buffer:
            #split buffer at the first newline
            line, self.buffer = self.buffer.split("\n", 1)
            
            #log the complete line if it's not empty
            if line.strip() != "":
                self.logger.log(self.level, line.strip())

    def flush(self):
        #flush remaining buffer if any
        if self.buffer.strip() != "":
            self.logger.log(self.level, self.buffer.strip())
            self.buffer = ""

#redirect stdout to our logger
sys.stdout = StreamToLogger(logger)

#decorator to log inputs and outputs silently
def log_execution(func):
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        
        #helper to truncate large arguments for readable logs
        def format_arg(arg):
            #handle pandas objects
            if isinstance(arg, (pd.DataFrame, pd.Series)):
                return f"<{type(arg).__name__} shape={arg.shape}>"
            
            #handle numpy arrays
            if isinstance(arg, np.ndarray):
                return f"<ndarray shape={arg.shape} dtype={arg.dtype}>"
            
            #handle large lists (e.g. loss history)
            if isinstance(arg, list) and len(arg) > 5:
                return f"<list len={len(arg)}>"
            
            #return string representation for small objects
            return repr(arg)

        #format inputs
        args_repr = [format_arg(a) for a in args]
        kwargs_repr = [f"{k}={format_arg(v)}" for k, v in kwargs.items()]
        
        #log function start with args
        try:
            logger.info(f"STARTED: {func_name} | Args: {args_repr} | Kwargs: {kwargs_repr}")
        except Exception:
            logger.info(f"STARTED: {func_name} | (Args could not be serialized)")

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            
            #format output/return value
            if isinstance(result, tuple):
                #truncate tuple elements individually
                ret_str = f"tuple({', '.join([format_arg(x) for x in result])})"
            else:
                ret_str = format_arg(result)
            
            #log completion time and result
            logger.info(f"FINISHED: {func_name} | Duration: {duration:.4f}s | Output: {ret_str}")
            return result
            
        except Exception as e:
            logger.error(f"EXCEPTION in {func_name}: {e}")
            raise e
            
    return wrapper

#seaborn themes
sns.set_theme(
    style="whitegrid",      # clean background
    palette="deep",         # modern default colors
    font_scale=1.2          # slightly larger text
)




#actual linear regression
@log_execution
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


@log_execution
def choose_dataset() -> None:
    #use sys.__stdout__ to ensure prompt is visible despite redirection
    print("Choose dataset 1 (success) or 2 (failure due to data):", file=sys.__stdout__)
    c = input() #read input from standard input
    
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
@log_execution
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
