from xmlrpc.client import Boolean
import os
from dateutil.relativedelta import relativedelta
import time
import requests
import pandas as pd


#TODO Place Stoploss

class Oanda():
        def __init__(self, demo=True):
            self.setEnv(demo)
            self.API_URL,self.ACCESS_TOKEN,self.ACCOUNT_ID = self.readLoginData()

        def setEnv(self,demo:Boolean):
            """Set if live or demo account

            Args:
                demo (Boolean): True if demo. False if live
            """
            if demo == True:
                self.env = "demo"
            else:
                self.env = "live"
                
        
        def readLoginData(self):
            """Read login data from text file

            Returns:
                loginData: Login data
            """
            dirname = os.path.dirname(__file__)
            file = os.path.join(dirname, "./LoginData/" + self.env + ".txt")
            loginData = []
            with open(file) as f:
                contents = f.readlines()  # API_URL # ACCESS_TOKEN # ACCOUNT_ID
            for line in contents:
                loginData.append(line.rstrip())
            return loginData[0],loginData[1],loginData[2]

      
        def json_to_pandas(json) -> pd.DataFrame:
            """Convert json to pandas

            Args:
                json (str): String in json format

            Returns:
                pd.dataframe: Pandas dataframe time series
            """
            json_file = json.json()
            price_json = json_file["candles"]
            times = []
            close_price, high_price, low_price, open_price = [], [], [], []

            for candle in price_json:
                times.append(candle["time"])
                close_price.append(float(candle["mid"]["c"]))
                high_price.append(float(candle["mid"]["h"]))
                low_price.append(float(candle["mid"]["l"]))
                open_price.append(float(candle["mid"]["o"]))

            dataframe = pd.DataFrame({"close": close_price, "high": high_price, "low": low_price, "open": open_price})
            dataframe.index = pd.to_datetime(times)
            return dataframe

        
        def getHistoricalData(self,symbol,start_date,end_date,timeframe) -> pd.DataFrame:
            """Get Historical Data

            Args:
                symbol (str): Instrument name
                start_date (str): Datetime
                end_date (str): Datetime
                timeframe (str, optional): timeframe. Defaults to "D".

            Returns:
                pd.DataFrame: time series
            """
            from_time = time.mktime((start_date).timetuple())
            to_time = time.mktime((end_date).timetuple())
            header = {"Authorization": "Bearer " + self.ACCESS_TOKEN}
            query = {"from": str(from_time), "to": str(to_time), "granularity": timeframe}
            CANDLES_PATH = f"/v3/accounts/{self.ACCOUNT_ID}/instruments/{symbol}/candles"
            response = requests.get("https://" + self.API_URL + CANDLES_PATH, headers=header, params=query)
            print(response)
            hist_df = Oanda.json_to_pandas(response)
            hist_df["Year"] = hist_df.index.year
            hist_df["Month"] = hist_df.index.month
            hist_df["Day"] = hist_df.index.day
            hist_df["Hour"] = hist_df.index.hour
            hist_df["Minute"] = hist_df.index.minute
            hist_df.reset_index(drop=True, inplace=True)
            return hist_df
        
      