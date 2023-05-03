import sys
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates

from Agent import QAgent
import Config

class Simulate:
    def loadData(self):
        # df_full = pd.read_csv("^GSPC_2010-2015.csv")
        predictedData = pd.read_csv("predictedPrice.csv",index_col=False)
        predictedData.set_index('Date')
        # predValue = predictedData.rename(columns={"Pred_Close": "Close"})
        # total = pd.concat(df_full,predValue)
        return predictedData
    
    def addFuturePricePrediction():
        pass

    def trainModel(self):
        self.data = self.loadData()
        self.close = self.data.Close.values.tolist()
        self.agent = QAgent(state_size = Config.window_size, 
                window_size = Config.window_size, 
                trend = self.close, 
                skip = Config.skip, 
                batch_size = Config.batch_size)
        self.agent.train(iterations = Config.iterations, checkpoint = Config.checkpoint, initial_money = Config.initial_money)


    def inferModel(self):
        name = "Q-Learning_Agent_Recommendation"
        
        states_buy, states_sell, total_gains, invest = self.agent.buy(Config.initial_money,self.close)
        fig = plt.figure(figsize = (15,5))
        plt.plot(self.close, color='b', lw=2.)
        plt.plot(self.close, '^', markersize=3, color='g', label = 'buying signal', markevery = states_buy)
        plt.plot(self.close, 'v', markersize=3, color='r', label = 'selling signal', markevery = states_sell)
        # plt.title('Profit %f, Returns %f%%'%(total_gains, invest))
        plt.axvspan(1511,1540, color='red', alpha=0.5)
        plt.title("Returns = {} by ({}%), Capital ={} [Predicted price trade suggestions highlighted in Red]".format(round(total_gains,2),round(invest,2),round(Config.initial_money + total_gains,2)))
        plt.legend()
        plt.savefig(name+'.png')
        plt.show()


if __name__ == "__main__":
    argList = sys.argv
    simulate = Simulate()
    if "simulate" in argList:
        simulate.trainModel()
        simulate.inferModel()
    elif "info" in argList:
        pass
    else:
        print("Please appropriate argument!!")