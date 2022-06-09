import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import pearsonr 

def upper_shadow(df): return df['High'] - np.maximum(df['Close'], df['Open'])
def lower_shadow(df): return np.minimum(df['Close'], df['Open']) - df['Low']

def get_features(df, row = False):
    df_feat = df
    df_feat['mean_trade'] = df_feat['Volume']/df_feat['Count']
    df_feat['upper_Shadow'] = upper_shadow(df_feat)
    df_feat['lower_Shadow'] = lower_shadow(df_feat)
    df_feat['shadow3'] = df_feat['upper_Shadow'] / df_feat['Volume']
    df_feat['shadow5'] = df_feat['lower_Shadow'] / df_feat['Volume']
    df_feat['mean1'] = (df_feat['shadow5'] + df_feat['shadow3']) / 2
    df_feat['UPS'] = (df_feat['High'] - np.maximum(df_feat['Close'], df_feat['Open']))
    df_feat['LOS'] = (np.minimum(df_feat['Close'], df_feat['Open']) - df_feat['Low'])
    df_feat['RNG'] = ((df_feat['High'] - df_feat['Low']) / df_feat['VWAP'])
    df_feat['MOV'] = ((df_feat['Close'] - df_feat['Open']) / df_feat['VWAP'])
    df_feat['LOGVOL'] = np.log(1. + df_feat['Volume'])
    df_feat['LOGCNT'] = np.log(1. + df_feat['Count'])
    if row: df_feat['Mean'] = df_feat[['Open', 'High', 'Low', 'Close']].mean()
    else: df_feat['Mean'] = df_feat[['Open', 'High', 'Low', 'Close']].mean(axis = 1)
    df_feat["High/Mean"] = df_feat["High"] / df_feat["Mean"]
    df_feat["Low/Mean"] = df_feat["Low"] / df_feat["Mean"]
    df_feat["Volume/Count"] = df_feat["Volume"] / (df_feat["Count"] + 1)
    mean_price = df_feat[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    df_feat['high2mean'] = df_feat['High'] / mean_price
    df_feat['low2mean'] = df_feat['Low'] / mean_price
    df_feat['volume2count'] = df_feat['Volume'] / (df_feat['Count'] + 1)
    print(df_feat)
    return df_feat

def decomposition_noise(df,period=60):
    df_feat = df
    result_mult=seasonal_decompose(df_feat['Close'], model='multiplicative', period=period)
    result_add=seasonal_decompose(np.log(df_feat['Close']), model='additive', period=period)
    df_feat["noise_mult"] = result_mult.resid
    df_feat["noise_add"] = result_add.resid
    return df_feat

def fft_feature(df):
    df_feat = df
    close_fft = np.fft.fft(np.asarray(df_feat['Close'].tolist()))
    fft_df = pd.DataFrame({'fft':close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [5, 10, 50]:
        fft_list= np.copy(fft_list); fft_list[num_:-num_]=0
        df_feat["fft_" + str(num_)] = np.fft.ifft(fft_list).real
    return df_feat
    

def tech_analysis(df):
    df_feat = df
    df_feat["MA_20"] = df_feat["Close"].rolling(20).mean()
    df_feat["MA_diff"] = np.log(1+(df_feat.MA_20 - df_feat.Close)/df_feat.Close.shift(1))
    
    df_feat["std_kurz"] = df_feat["Close"].rolling(60).std()
    df_feat["mean_kurz"] = df_feat["Close"].rolling(60).mean()


    df_feat['Bollinger_High_urban'] = df_feat["mean_kurz"] + (df_feat["std_kurz"] * 3.5)
    df_feat ['Bollinger_Low_urban'] = df_feat["mean_kurz"] - (df_feat["std_kurz"] * 0.5) 
    return df_feat
    
def get_macd(price, slow, fast, smooth):
    exp1 = price.ewm(span = fast, adjust = False).mean()
    exp2 = price.ewm(span = slow, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns = {'Close':'macd'})
    print(macd)
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'macd':'signal'})
    hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns = {0:'hist'})
    frames =  [macd, signal, hist]
    df = pd.concat(frames, join = 'inner', axis = 1)
    return df

def outlier_correction(df):
    df_feat = df
    df_feat = np.log(1+(df_feat["Close"]-df_feat["Close"].shift(1))/df_feat["Close"].shift(1))
    df_feat = detect_outliers(df_feat)
    
    
#https://github.com/krishnaik06/Finding-an-Outlier/blob/master/Finding%20an%20outlier%20in%20a%20Dataset.ipynb
def detect_outliers(data,std_threshold=60):  
    row_names = ["High","Open","Low","Close"]
    df = data
    for column in row_names:
        log_column_name = "Log_" + column
        df[log_column_name] = np.log(1+(df[column]-df[column].shift(1))/df[column].shift(1))
        # threshold=std_threshold
        # mean = np.mean(df[log_column_name])
        # std =np.std(df[log_column_name])
        # counter = 0
        # for index, row in df.iterrows():
        #     if not (row[log_column_name] != row[log_column_name]):
        #         z_score = (row[log_column_name] - mean)/std 
        #         if z_score > threshold:
        #             df.at[index, log_column_name] = (threshold*std)+mean
        #             counter += 1
        #         elif z_score < threshold*-1:
        #             df.at[index, log_column_name] = (-1*threshold*std)+mean
        #             counter += 1
    df = df.dropna()
    return df

def feature_seasonal(df):
    df_feat = df
    df_feat["Minute"] = df_feat.index.minute
    df_feat["Day"] = df_feat.index.day
    df_feat["Month"] = df_feat.index.month
    df_feat["Year"] = df_feat.index.year
    df_feat["DayOfWeek"] = df_feat.index.dayofweek
    btc_season_log = df_feat["Log_Close"]
    btc_seasonal_min =  btc_season_log.groupby(btc_season_log.index.minute).median()
    df_feat["minute_seasonal"] = df_feat.Minute.apply(min_mapper, pd_series = btc_seasonal_min)
    def get_correlation(vals):
        return pearsonr(vals, btc_seasonal_min)[0]
    df_feat['correlation_log_perf'] = btc_season_log.rolling(window=len(btc_seasonal_min)).apply(get_correlation)
    return df_feat
    


def min_mapper(value,pd_series):
    return pd_series.loc[value]


def createFeature(df):
    df_1 = get_features(df)
    print("Step 1")
    df_2 = decomposition_noise(df_1)
    print("Step 2")
    df_3 = fft_feature(df_2)
    print("Step 3")
    df_4 = tech_analysis(df_3)
    print("Step 4")
    df_5 = df_4.join(get_macd(df_4['Close'], 26, 12, 9))
    df_5 = df_4.dropna()
    return df_5
    
    # TODO Outlier correction
    

data_path = r"C:\\Users\\Armin\\Documents\\Code\\Studienarbeit\\crypto_daten_kaggle\\"
crypto_df = pd.read_hdf("train.h5")
for i in range(0,14):
    print(i)
    btc = crypto_df[crypto_df["Asset_ID"]==i]
    btc["timestamp"] = pd.to_datetime(btc["timestamp"], unit='s')
    btc.set_index("timestamp",inplace=True)

    btc = createFeature(btc)
    btc = btc.dropna()
    btc = detect_outliers(btc)
    btc = feature_seasonal(btc)
    btc = btc.dropna()
    btc_target_label = []
    for index,value in btc["Target"].items():
        if value < 0:
            btc_target_label.append(1)
        elif value > 0:
            btc_target_label.append(0)
        else:
            btc_target_label.append(0)
    btc["Label"] = np.array(btc_target_label)
    btc = btc.drop(["Target","Asset_ID","index","Open","High","Low","Close"], axis=1).reset_index(drop=True)
    btc.to_hdf('Data/' + str(i) + '_V1.h5', key='df', mode='w')
