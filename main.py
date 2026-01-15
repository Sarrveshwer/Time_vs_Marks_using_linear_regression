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
    a=2 #int(input("Test(1) or Not (2): "))
    if a==1:
        os.mkdir("logs")
    else:
        os.mkdir("logs_test")
    os.mkdir("images")
except FileExistsError:
    pass
except OSError as e:
    print(f"An error occurred: {e}")


safe_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if a==1:
    log_path = os.path.join("logs", f"{filename}@{safe_timestamp}.log")
else:
    log_path = os.path.join("logs_test", f"{filename}_test@{safe_timestamp}.log")

sys.stdout = Logger(log_path)


def input_and_log(prompt=""):
    print(prompt, end="", flush=True)
    answer = real_input()
    if not sys.stdout.log.closed:
        sys.stdout.log.write(answer + "\n")
        sys.stdout.log.flush()
    return answer


input = input_and_log

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    
    print("\n" + "="*30 + " CRASH DETECTED " + "="*30)
    print(f"Timestamp: {datetime.datetime.now()}")
    print(error_msg)
    print("="*76 + "\n")

sys.excepthook = handle_exception

#actual linear regression

class LinearRegressionModel:
    def __init__(self):
        self.x = None
        self.y = None
        self.x_name = None
        self.y_name = None
        self.data = None
        self.w = 0.0
        self.b = 0.0
        self.losses = []
        self.ws = []
        self.bs = []

    def load_data(self, choice):
        if choice == "1":
            self.data = pd.read_csv('dataset_study.csv', usecols=[1,2])
            print(self.data)
            self.x = self.data["study_hours"]
            self.y = self.data['grade']
            self.x_name = "study_hours"
            self.y_name = 'grade'
        else:
            self.data = pd.read_csv('dataset.csv', usecols=[9,16], names=["Score","Hours"], skiprows=1)
            print(self.data)
            self.x = self.data["Hours"].astype('float64')
            self.y = self.data['Score'].astype('float64')
            self.x_name = 'Hours'
            self.y_name = 'Score'

    def train_step(self, x, y, lr, epochs):
        w = 0.0
        b = 0.0
        n = len(x)
        
        ws, bs, losses = [], [], []
        for i in range(epochs):
            y_pred = b + w*x
            error = y - y_pred
            #calculate loss
            loss = np.mean(error**2)
            losses.append(loss)
            #update and change weight and bias
            ws.append(w)
            bs.append(b)
            dw = -2/n * np.sum(x*error)
            db = -2/n * np.sum(error)
            
            w -= lr*dw
            b -= lr*db
        
        #Slope = weight
        slope = w
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
        residuals = y - y_pred
        ssr = np.sum(residuals**2)
        s2 = ssr/(n-2)
        
        std_err = np.sqrt(s2/ss_xx)
        
        #p-value calculation using scipy
        #NOTE: p-value is gives if the there is an actual relation between the two variables x and y
        #FORUMULA IS 
        t_stat = slope / std_err
        df = n - 2
        p_value = 2 * (1 - t.cdf(abs(t_stat), df=df))
        
        return w, b, correlation, p_value, std_err, losses, ws, bs

    def run(self, dataset_choice):
        self.load_data(dataset_choice)
        
        #normalize data to try to imrpove performance and p-value   
        x_mean = self.x.mean()
        x_std = self.x.std()
        x_norm = (self.x - x_mean) / x_std

        y_mean = self.y.mean()
        y_std = self.y.std()
        y_norm = (self.y - y_mean) / y_std
        
        mse_best = 100000000000
        best_lr = 0
        training_epochs = 200
        
        for lr in [0.1, 0.01, 0.001, 0.0001]:
            *_, losses_list, _, _ = self.train_step(x_norm, y_norm, lr, training_epochs)
            final_loss = losses_list[-1]
            if final_loss < mse_best:
                mse_best = final_loss
                best_lr = lr
                
        print("Best lr=", best_lr)
        slope, intercept, correlation, p_value, std_err, losses_list, weights_list, biases_list = \
            self.train_step(x_norm, y_norm, best_lr, training_epochs)

        slope = slope * (y_std / x_std)
        intercept = y_mean + y_std * intercept - slope * x_mean

        y_pred = slope * self.x + intercept 
        mse = np.mean((self.y - y_pred) ** 2) 
        rmse = np.sqrt(mse) 
        print("MSE=", mse)
        print("RMSE=", rmse)
        if correlation < 0.70:
            print("Correlation=", correlation)
        else:
            print("Correlation=", correlation)

        alpha = 0.07
        if p_value <= alpha:
            print("Slope is statistically significant")
        else:
            print("No statistically significant linear relationship")
        print("p-value=", p_value)

        x_line = np.linspace(min(self.x), max(self.x), 100)
        y_line = slope * x_line + intercept

        layout = [
            ["loss", "params"],
            ["scatter", "scatter"]
        ]

        safe_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
        sns.scatterplot(data=self.data, x=self.x_name, y=self.y_name, color="red", ax=axes["scatter"])
        axes["scatter"].plot(x_line, y_line, color="blue", label='slope')
        axes["scatter"].set_title("Linear Regression")
        axes["scatter"].legend()

        # Adjust spacing to prevent overlap
        plt.tight_layout()
        plt.savefig(os.path.join("images", f"Output_{safe_timestamp}.png"), dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    print("\n--- Running Dataset 1 (Synthetic) ---")
    model1 = LinearRegressionModel()
    model1.run("1")

    print("\n--- Running Dataset 2 (Real Case) ---")
    model2 = LinearRegressionModel()
    model2.run("2")
