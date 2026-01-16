import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
from scipy.interpolate import interpn
import logging
import time
import sys
import datetime
import os
import traceback


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
        self.slope = 0.0
        self.intercept = 0.0
        self.y_pred = None
        self.correlation = 0.0
        self.p_value = 0.0
        self.std_err = 0.0
        self.mse = 0.0
        self.r_squared = 0.0

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
        self.slope, self.intercept, self.correlation, self.p_value, self.std_err, self.losses, self.ws, self.bs = \
            self.train_step(x_norm, y_norm, best_lr, training_epochs)

        self.slope = self.slope * (y_std / x_std)
        self.intercept = y_mean + y_std * self.intercept - self.slope * x_mean

        self.y_pred = self.slope * self.x + self.intercept 
        self.mse = np.mean((self.y - self.y_pred) ** 2) 
        rmse = np.sqrt(self.mse) 
        
        # Calculate R-squared
        ss_res = np.sum((self.y - self.y_pred) ** 2)
        ss_tot = np.sum((self.y - self.y.mean()) ** 2)
        self.r_squared = 1 - (ss_res / ss_tot)

        print("MSE=", self.mse)
        print("RMSE=", rmse)
        print("Correlation (R)=", self.correlation)
        print("R-squared=", self.r_squared)

        alpha = 0.07
        if self.p_value <= alpha:
            print("Slope is statistically significant")
        else:
            print("No statistically significant linear relationship")
        print("p-value=", self.p_value)

        self.plot()

    def plot(self):
        plt.rcParams.update({'font.size': 12, 'axes.titlesize': 16, 'axes.labelsize': 14})
        sns.set_theme(style="darkgrid")
        safe_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        layout = [
            ["Residual", "loss"],
            ["params", "residuals"]
        ]
        
        fig, axes = plt.subplot_mosaic(layout, figsize=(16, 14), constrained_layout=True)

        # 1) Residual Plot with Distance-Based Coloring (Closer = Darker)
        x_plot = self.x
        y_plot = self.y
        residuals = self.y - self.y_pred
        r = np.abs(residuals)
        
        sns.scatterplot(x=self.y_pred,
                y=residuals, 
                hue=r,
                palette="mako",
                alpha=0.7,
                edgecolor="none",   
                linewidth=0,
                ax=axes["Residual"])
        axes["Residual"].set_title("Residual Plot")
        axes["Residual"].set_xlabel("Residuals")
        axes["Residual"].set_ylabel("Predicted Values")
        axes["Residual"].axhline(y=0, color='#41b5ac', linewidth=3,alpha=0.9)
        

        # 2) Loss Curve Plot
        epochs = range(len(self.losses))
        axes["loss"].plot(epochs, self.losses, color='#FF4500', linewidth=3)
        
        min_loss = min(self.losses)
        axes["loss"].fill_between(epochs, self.losses, min_loss, color='#FF4500', alpha=0.3)
        
        axes["loss"].set_title("Model Optimization Convergence", fontweight='bold')
        axes["loss"].set_xlabel("Epochs")
        axes["loss"].set_ylabel("MSE (Normalized)")
        
        stats_text = (f'Final MSE (Raw): {self.mse:.4f}\n'
                      f'Correlation (R): {self.correlation:.4f}\n'
                      f'R² Score: {self.r_squared:.4f}')
        
        axes["loss"].text(0.95, 0.95, stats_text, transform=axes["loss"].transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       fontsize=12, fontweight='bold',
                       bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', boxstyle='round,pad=1'), color='white')

        # 3) Parameters Evolution
        axes["params"].plot(self.ws, color='#1f77b4', linewidth=2.5, label="Slope (w)")
        axes["params"].plot(self.bs, color='#ff7f0e', linewidth=2.5, label="Intercept (b)")
        
        marker_interval = max(1, len(self.ws) // 10)
        axes["params"].plot(range(0, len(self.ws), marker_interval), 
                         [self.ws[i] for i in range(0, len(self.ws), marker_interval)], 
                         'o', color='#1f77b4', markersize=6)
        axes["params"].plot(range(0, len(self.bs), marker_interval), 
                         [self.bs[i] for i in range(0, len(self.bs), marker_interval)], 
                         's', color='#ff7f0e', markersize=6)

        axes["params"].set_title("Parameters Evolution", fontweight='bold')
        axes["params"].set_xlabel("Iteration")
        axes["params"].set_ylabel("Normalized Value")
        axes["params"].legend()

        # 4) Residual Distribution
        sns.histplot(residuals, kde=True, ax=axes["residuals"], color="purple", alpha=0.4)
        axes["residuals"].set_title("Residual Distribution (Normality Check)", fontweight='bold')
        axes["residuals"].set_xlabel("Residual Value")
        axes["residuals"].set_ylabel("Frequency")

        plt.suptitle(f"Comprehensive Analysis Dashboard: {self.y_name} vs {self.x_name}", fontsize=22, fontweight='bold', y=0.98)
        
        save_path = os.path.join("images", f"Dashboard_{safe_timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Polished dashboard saved to {save_path}")
        plt.show()

if __name__ == "__main__":
    print("\n--- Running Dataset 1 (Synthetic) ---")
    model1 = LinearRegressionModel()
    model1.run("1")

    print("\n--- Running Dataset 2 (Real Case) ---")
    model2 = LinearRegressionModel()
    model2.run("2")
