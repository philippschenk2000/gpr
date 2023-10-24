import pandas as pd
import numpy as np
import random as rnd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os

# machine learning
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor


import indicators.bb as bb
import indicators.ema200_rsi as ema200_rsi
import indicators.rsi as rsi
import math
import os
import patterns.cross as cross
import patterns.cup_handle as cup_handle
import patterns.flag as flag
import patterns.fla as fla
import patterns.flb as flb
import patterns.flc as flc
import patterns.fld as fld
import patterns.ret300 as ret300
import patterns.ret100 as ret100
import patterns.mret200 as mret200
import patterns.mret100 as mret100
import patterns.fiba as fiba
import patterns.fibb as fibb
import patterns.fibc as fibc
import patterns.wedge as wedge
import patterns.weaka as weaka
import patterns.weakb as weakb
import patterns.weakc as weakc
import patterns.weakd as weakd
import patterns.kkerze as kkerze
import patterns.ukerze as ukerze
import preprocessing.highs as highs
import preprocessing.highshighs as highshighs
import preprocessing.hocs as hocs
import preprocessing.lows as lows
import preprocessing.lowslows as lowslows
import preprocessing.locs as locs
import preprocessing.hors as hors
import preprocessing.trends as trends
import preprocessing.FLpos as FLpos
import preprocessing.FLneg as FLneg
import requests
import time
import heapq


def main():
    warnings.filterwarnings("ignore")
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    symbol = input("Enter your symbol (aapl, msft):")
    tage = 1000 #int(input("How many days you want to backtest? (1-3000):"))
    length = 2 # int(input("How exact you want to test? (1-6):"))
    print('Signale werden berechnet...')
    candlesticks = get_candlesticks(symbol, tage)
    candlesticks.reverse()
    gefiltert = pd.DataFrame()

    events = calculate(candlesticks, tage, length)
    events0 = filteringbuy(events)
    events2 = filteringsell(events)

    backtest0 = backtesting0(events0, symbol)
    backtest2 = backtesting2(events2, symbol)
    gefiltert = backtest0.append(backtest2)
    with open('1allevents.textmate', 'w') as file:
        file.write(str(events) + '\n')

    backtest = backtesting3(gefiltert, tage, events, length, symbol)
    speicher = abspeichern(gefiltert, tage, symbol, backtest)


def get_candlesticks(symbol, tage):
    url = "https://cloud.iexapis.com/stable/stock/" + symbol + "/chart/max"
    r = requests.get(url, params=payload)
    r.raise_for_status()
    prices = r.json()
    for price in prices:
        for key in list(price):
            if key not in ["date", "high", "low", "open", "close"]:
                del price[key]
    return prices


def calculate(candlesticks, tage, length):
    events = pd.DataFrame()
    event = pd.DataFrame()
    for i in range(0, tage):
        prices = []
        interval = candlesticks[i:i+300]
        for candlestick in interval:
            prices.append([float(candlestick['open']), float(candlestick['high']), float(candlestick['low']), float(candlestick['close'])])
        step = math.floor(((prices[-1][1] - prices[-1][2]) / 3) * 1000) / 1000
        prices[-1][3] = prices[-1][2]
        df1 = [];df2 = [];df3 = [];df4 = [];df5 = [];df6 = [];df7 = [];df8 = [];df9 = [];df10 = []
        df11 = [];df12 = [];df13 = [];df14 = [];df15 = [];df16 = [];df17 = [];df18 = [];df19 = [];df20 = []
        z = i/int(tage)

        print('Fortschritt: ' + str(round(z*100, 1)) + '%')

        for i in range(0, int(length)):

            result = get_events(prices)
            event['date'] = [interval[-1]["date"]]
            event['price'] = [round(prices[-1][3], 2)]
            event['rsi'] = result['rsi']
            event['ema'] = result['ema']
            event['bb'] = result['bb']
            event['fla'] = result['fla']
            event['flb'] = result['flb']
            event['flc'] = result['flc']
            event['fld'] = result['fld']
            event['weaka'] = result['weaka']
            event['weakb'] = result['weakb']
            event['weakc'] = result['weakc']
            event['weakd'] = result['weakd']
            event['cross'] = result['cross']
            event['h4'] = result['h4']
            event['h5'] = result['h5']
            event['h6'] = result['h6']
            event['ret3'] = result['ret3']
            event['ret1'] = result['ret1']
            event['mret2'] = result['mret2']
            event['mret1'] = result['mret1']
            event['fiba'] = result['fiba']
            event['fibb'] = result['fibb']
            event['fibc'] = result['fibc']
            event['kk'] = result['kk']
            event['uk'] = result['uk']

            events = events.append(event, ignore_index=True)
            prices[-1][3] += step
    corr = correlation(events)

    x = 0
    for i in range(0, tage*int(length) - 20):
        df1.append(float(events['price'].loc[x+1 : x+1]))
        df2.append(float(events['price'].loc[x+2 : x+2]))
        df3.append(float(events['price'].loc[x+3 : x+3]))
        df4.append(float(events['price'].loc[x+4 : x+4]))
        df5.append(float(events['price'].loc[x+5 : x+5]))
        df6.append(float(events['price'].loc[x+6 : x+6]))
        df7.append(float(events['price'].loc[x+7 : x+7]))
        df8.append(float(events['price'].loc[x+8 : x+8]))
        df9.append(float(events['price'].loc[x+9 : x+9]))
        df10.append(float(events['price'].loc[x+10 : x+10]))
        df11.append(float(events['price'].loc[x+11 : x+11]))
        df12.append(float(events['price'].loc[x+12 : x+12]))
        df13.append(float(events['price'].loc[x+13 : x+13]))
        df14.append(float(events['price'].loc[x+14 : x+14]))
        df15.append(float(events['price'].loc[x+15 : x+15]))
        df16.append(float(events['price'].loc[x+16 : x+16]))
        df17.append(float(events['price'].loc[x+17 : x+17]))
        df18.append(float(events['price'].loc[x+18 : x+18]))
        df19.append(float(events['price'].loc[x+19 : x+19]))
        df20.append(float(events['price'].loc[x+20 : x+20]))

        x = x+1
    for i in range(0, 20):
        df1.append(float(events['price'].loc[x : x]))
        df2.append(float(events['price'].loc[x : x]))
        df3.append(float(events['price'].loc[x : x]))
        df4.append(float(events['price'].loc[x : x]))
        df5.append(float(events['price'].loc[x : x]))
        df6.append(float(events['price'].loc[x : x]))
        df7.append(float(events['price'].loc[x : x]))
        df8.append(float(events['price'].loc[x : x]))
        df9.append(float(events['price'].loc[x : x]))
        df10.append(float(events['price'].loc[x : x]))
        df11.append(float(events['price'].loc[x : x]))
        df12.append(float(events['price'].loc[x : x]))
        df13.append(float(events['price'].loc[x : x]))
        df14.append(float(events['price'].loc[x : x]))
        df15.append(float(events['price'].loc[x : x]))
        df16.append(float(events['price'].loc[x : x]))
        df17.append(float(events['price'].loc[x : x]))
        df18.append(float(events['price'].loc[x : x]))
        df19.append(float(events['price'].loc[x : x]))
        df20.append(float(events['price'].loc[x : x]))

    events['price+1'] = df1
    events['price+2'] = df2
    events['price+3'] = df3
    events['price+4'] = df4
    events['price+5'] = df5
    events['price+6'] = df6
    events['price+7'] = df7
    events['price+8'] = df8
    events['price+9'] = df9
    events['price+10'] = df10
    events['price+11'] = df11
    events['price+12'] = df12
    events['price+13'] = df13
    events['price+14'] = df14
    events['price+15'] = df15
    events['price+16'] = df16
    events['price+17'] = df17
    events['price+18'] = df18
    events['price+19'] = df19
    events['price+20'] = df20

    return events


def get_events(prices):

    highs_level_two = highs.get_level_two(prices)
    highs_level_three = highs.get_level_three(prices)
    highs_level_four = highs.get_level_four(prices)
    highs_level_five = highs.get_level_five(prices)
    lows_level_two = lows.get_level_two(prices)
    lows_level_three = lows.get_level_three(prices)
    lows_level_four = lows.get_level_four(prices)
    lows_level_five = lows.get_level_five(prices)
    lines_high = trends.lines_high(prices, highs_level_three, highs_level_four, highs_level_five)
    lines_low = trends.lines_low(prices, lows_level_three, lows_level_four, lows_level_five)

    highshighs_level_three = highshighs.get_level_three(prices)
    highshighs_level_four = highshighs.get_level_four(prices)
    highshighs_level_five = highshighs.get_level_five(prices)
    highshighs_level_six = highshighs.get_level_six(prices)
    lowslows_level_one = lowslows.get_level_one(prices)
    lowslows_level_two = lowslows.get_level_two(prices)
    lowslows_level_three = lowslows.get_level_three(prices)
    lowslows_level_four = lowslows.get_level_four(prices)
    lowslows_level_five = lowslows.get_level_five(prices)
    lowslows_level_six = lowslows.get_level_six(prices)
    hocs_level_four = hocs.get_level_four(prices)
    hocs_level_five = hocs.get_level_five(prices)
    hocs_level_six = hocs.get_level_six(prices)
    locs_level_four = locs.get_level_four(prices)
    locs_level_five = locs.get_level_five(prices)
    locs_level_six = locs.get_level_six(prices)
    horizontal_level_four = hors.horizontal_level_four(prices, highshighs_level_four, lowslows_level_four)
    horizontal_level_five = hors.horizontal_level_five(prices, highshighs_level_five, lowslows_level_five)
    horizontal_level_six = hors.horizontal_level_six(prices, highshighs_level_six, lowslows_level_six)

    result = pd.DataFrame()
    result['rsi'] = [0 if rsi.buy(prices) and FLneg.ja(prices) else 1]
    result['ema'] = [0 if ema200_rsi.buy(prices) and FLneg.ja(prices) else (2 if ema200_rsi.sell(prices) and FLpos.ja(prices) else 1)]
    result['bb'] = [0 if bb.buy(prices) else 1]
    result['fla'] = [0 if fla.buy(prices) and FLneg.ja(prices) else 1]
    result['flb'] = [0 if flb.buy(prices) and FLneg.ja(prices) else 1]
    result['flc'] = [0 if flc.buy(prices) and FLneg.ja(prices) else 1]
    result['fld'] = [0 if fld.buy(prices) and FLneg.ja(prices) else 1]
    result['weaka'] = [2 if weaka.sell(prices) and FLpos.ja(prices) else 1]
    result['weakb'] = [2 if weakb.sell(prices) and FLpos.ja(prices) else 1]
    result['weakc'] = [2 if weakc.sell(prices) and FLpos.ja(prices) else 1]
    result['weakd'] = [2 if weakd.sell(prices) and FLpos.ja(prices) else 1]
    result['cross'] = [0 if cross.buy(prices, lines_low, horizontal_level_four) and FLneg.ja(prices)  else (2 if cross.sell(prices, lines_high, horizontal_level_four) and FLpos.ja(prices) else 1)]
    result['h4'] = [0 if hors.horizontal4_buy(prices, horizontal_level_four) and FLneg.ja(prices) else (2 if hors.horizontal4_sell(prices, horizontal_level_four) and FLpos.ja(prices) else 1)]
    result['h5'] = [0 if hors.horizontal5_buy(prices, horizontal_level_five) and FLneg.ja(prices) else (2 if hors.horizontal5_sell(prices, horizontal_level_five) and FLpos.ja(prices) else 1)]
    result['h6'] = [0 if hors.horizontal6_buy(prices, horizontal_level_six) and FLneg.ja(prices) else (2 if hors.horizontal6_sell(prices, horizontal_level_six) and FLpos.ja(prices) else 1)]
    result['ret3'] = [0 if ret300.buy(prices, hocs_level_five) and FLneg.ja(prices) else (2 if ret300.sell(prices, locs_level_five) and FLpos.ja(prices) else 1)]
    result['ret1'] = [0 if ret100.buy(prices, hocs_level_four) and FLneg.ja(prices) else (2 if ret100.sell(prices, locs_level_four) and FLpos.ja(prices) else 1)]
    result['mret2'] = [0 if mret200.buy(prices, highshighs_level_five) and FLneg.ja(prices) else (2 if mret200.sell(prices, lowslows_level_five) and FLpos.ja(prices) else 1)]
    result['mret1'] = [0 if mret100.buy(prices, highshighs_level_four) and FLneg.ja(prices) else (2 if ret100.sell(prices, locs_level_four) and FLpos.ja(prices) else 1)]
    result['fiba'] = [0 if fiba.buy(prices, highshighs_level_five, lowslows_level_five) and FLneg.ja(prices) else 1]
    result['fibb'] = [0 if fibb.buy(prices, highshighs_level_four, lowslows_level_four) and FLneg.ja(prices) else 1]
    result['fibc'] = [0 if fibc.buy(prices, highshighs_level_three, lowslows_level_three) and FLneg.ja(prices) else 1]
    result['kk'] = [0 if kkerze.buy(prices) and FLneg.ja(prices) else 1]
    result['uk'] = [0 if ukerze.buy(prices) and FLneg.ja(prices) else (2 if ukerze.sell(prices) and FLpos.ja(prices) else 1)]

    return result


def filteringbuy(events):
    print('Performance wird berechnet...')
    buy = 0
    gefiltert = events.loc[(events['rsi'] == buy) | (events['ema'] == buy) | (events['bb'] == buy) | (events['fla'] == buy) | (
            events['flb'] == 0) | (events['flc'] == buy) | (events['fld'] == buy) | (events['weaka'] == buy) | (
                                   events['weakb'] == buy) | (events['weakc'] == buy) | (events['weakd'] == buy) | (
                                   events['cross'] == buy) | (events['h4'] == buy) | (events['h5'] == buy) | (
                                   events['h6'] == buy) | (events['ret3'] == buy) | (events['ret1'] == buy) | (
                                   events['mret1'] == buy) | (events['mret2'] == buy) | (events['fiba'] == buy) | (
                                   events['fibb'] == buy) | (events['fibc'] == buy) | (events['kk'] == buy) | (
                                   events['uk'] == buy)]
    gefiltert = gefiltert.replace(1, '')
    gefiltert = gefiltert.replace(0, 'buy')

    return gefiltert


def filteringsell(events):
    buy = 2
    gefiltert = events.loc[(events['rsi'] == buy) | (events['ema'] == buy) | (events['bb'] == buy) | (events['fla'] == buy) | (
            events['flb'] == 0) | (events['flc'] == buy) | (events['fld'] == buy) | (events['weaka'] == buy) | (
                                   events['weakb'] == buy) | (events['weakc'] == buy) | (events['weakd'] == buy) | (
                                   events['cross'] == buy) | (events['h4'] == buy) | (events['h5'] == buy) | (
                                   events['h6'] == buy) | (events['ret3'] == buy) | (events['ret1'] == buy) | (
                                   events['mret1'] == buy) | (events['mret2'] == buy) | (events['fiba'] == buy) | (
                                   events['fibb'] == buy) | (events['fibc'] == buy) | (events['kk'] == buy) | (
                                   events['uk'] == buy)]
    gefiltert = gefiltert.replace(1, '')
    gefiltert = gefiltert.replace(2, 'sell')

    return gefiltert


def backtesting0(events0, symbol):
    events0['perf+1'] = events0['price+1']/events0['price']
    events0['perf+2'] = events0['price+2']/events0['price']
    events0['perf+3'] = events0['price+3']/events0['price']
    events0['perf+4'] = events0['price+4']/events0['price']
    events0['perf+5'] = events0['price+5']/events0['price']
    events0['perf+6'] = events0['price+6']/events0['price']
    events0['perf+7'] = events0['price+7']/events0['price']
    events0['perf+8'] = events0['price+8']/events0['price']
    events0['perf+9'] = events0['price+9']/events0['price']
    events0['perf+10'] = events0['price+10']/events0['price']
    events0['perf+11'] = events0['price+11']/events0['price']
    events0['perf+12'] = events0['price+12']/events0['price']
    events0['perf+13'] = events0['price+13']/events0['price']
    events0['perf+14'] = events0['price+14']/events0['price']
    events0['perf+15'] = events0['price+15']/events0['price']
    events0['perf+16'] = events0['price+16']/events0['price']
    events0['perf+17'] = events0['price+17']/events0['price']
    events0['perf+18'] = events0['price+18']/events0['price']
    events0['perf+19'] = events0['price+19']/events0['price']
    events0['perf+20'] = events0['price+20']/events0['price']

    del events0['price+1'], events0['price+2'], events0['price+3'], events0['price+4'], events0['price+5'], events0['price+6'], events0['price+7'], events0['price+8'], events0['price+9'], events0['price+10'], events0['price+11'], events0['price+12'], events0['price+13'], events0['price+14'], events0['price+15'], events0['price+16'], events0['price+17'], events0['price+18'], events0['price+19'], events0['price+20']
    with open(symbol + '-buy' + '.textmate', 'w') as file:
        file.write(str(events0) + '\n')

    return events0


def backtesting2(events2, symbol):
    events2['perf+1'] = events2['price']/events2['price+1']
    events2['perf+2'] = events2['price']/events2['price+2']
    events2['perf+3'] = events2['price']/events2['price+3']
    events2['perf+4'] = events2['price']/events2['price+4']
    events2['perf+5'] = events2['price']/events2['price+5']
    events2['perf+6'] = events2['price']/events2['price+6']
    events2['perf+7'] = events2['price']/events2['price+7']
    events2['perf+8'] = events2['price']/events2['price+8']
    events2['perf+9'] = events2['price']/events2['price+9']
    events2['perf+10'] = events2['price']/events2['price+10']
    events2['perf+11'] = events2['price']/events2['price+11']
    events2['perf+12'] = events2['price']/events2['price+12']
    events2['perf+13'] = events2['price']/events2['price+13']
    events2['perf+14'] = events2['price']/events2['price+14']
    events2['perf+15'] = events2['price']/events2['price+15']
    events2['perf+16'] = events2['price']/events2['price+16']
    events2['perf+17'] = events2['price']/events2['price+17']
    events2['perf+18'] = events2['price']/events2['price+18']
    events2['perf+19'] = events2['price']/events2['price+19']
    events2['perf+20'] = events2['price']/events2['price+20']

    del events2['price+1'], events2['price+2'], events2['price+3'], events2['price+4'], events2['price+5'], events2['price+6'], events2['price+7'], events2['price+8'], events2['price+9'], events2['price+10'], events2['price+11'], events2['price+12'], events2['price+13'], events2['price+14'], events2['price+15'], events2['price+16'], events2['price+17'], events2['price+18'], events2['price+19'], events2['price+20']
    with open(symbol + '-sell' + '.textmate', 'w') as file:
        file.write(str(events2) + '\n')

    return events2


def backtesting3(gefiltert, tage, events, lentgh, symbol):

    a = float(events['price'].loc[0:0])
    b = float(events['price'].loc[tage*lentgh-1:tage*lentgh-1])

    df_perf = pd.DataFrame()
    df_perf['Performance der Signale in %'] = [round((gefiltert['perf+1'].mean()-1)*100, 2), round((gefiltert['perf+2'].mean()-1)*100, 2), round((gefiltert['perf+3'].mean()-1)*100, 2), round((gefiltert['perf+4'].mean()-1)*100, 2), round((gefiltert['perf+5'].mean()-1)*100, 2), round((gefiltert['perf+6'].mean()-1)*100, 2), round((gefiltert['perf+7'].mean()-1)*100, 2), round((gefiltert['perf+8'].mean()-1)*100, 2), round((gefiltert['perf+9'].mean()-1)*100, 2), round((gefiltert['perf+10'].mean()-1)*100, 2), round((gefiltert['perf+11'].mean()-1)*100, 2), round((gefiltert['perf+12'].mean()-1)*100, 2), round((gefiltert['perf+13'].mean()-1)*100, 2), round((gefiltert['perf+14'].mean()-1)*100, 2), round((gefiltert['perf+15'].mean()-1)*100, 2), round((gefiltert['perf+16'].mean()-1)*100, 2), round((gefiltert['perf+17'].mean()-1)*100, 2), round((gefiltert['perf+18'].mean()-1)*100, 2), round((gefiltert['perf+19'].mean()-1)*100, 2), round((gefiltert['perf+20'].mean()-1)*100, 2)]
    df_perf['Performance buy & hold in %'] = [round((b/a-1)*100/tage*1, 2), round((b/a-1)*100/tage*2, 2), round((b/a-1)*100/tage*3, 2), round((b/a-1)*100/tage*4, 2), round((b/a-1)*100/tage*5, 2), round((b/a-1)*100/tage*6, 2), round((b/a-1)*100/tage*7, 2), round((b/a-1)*100/tage*8, 2), round((b/a-1)*100/tage*9, 2), round((b/a-1)*100/tage*10, 2), round((b/a-1)*100/tage*11, 2), round((b/a-1)*100/tage*12, 2), round((b/a-1)*100/tage*13, 2), round((b/a-1)*100/tage*14, 2), round((b/a-1)*100/tage*15, 2), round((b/a-1)*100/tage*16, 2), round((b/a-1)*100/tage*17, 2), round((b/a-1)*100/tage*18, 2), round((b/a-1)*100/tage*19, 2), round((b/a-1)*100/tage*20, 2)]
    df_perf.plot(kind='line')
    plt.title('Performance results ' + symbol, fontsize=16)
    plt.show()

    return df_perf


def correlation(events):
    corr_mlr = events.corr(method='pearson')
    plt.figure(figsize=(20, 6))
    sns.heatmap(corr_mlr, annot=True, cmap='Blues')
    plt.title('Correlation matrix')
    plt.show()

    return corr_mlr#


def abspeichern(gefiltert, tage, symbol, backtest):

    df = pd.read_csv('Mappe2.csv', delimiter=' ')

    df = df.append({
            'symbol': symbol,
            'tage': tage,
            'perf+1': float(backtest['Performance der Signale in %'].loc[0 : 0]),
            'perf+2': float(backtest['Performance der Signale in %'].loc[1: 1]),
            'perf+3': float(backtest['Performance der Signale in %'].loc[2: 2]),
            'perf+4': float(backtest['Performance der Signale in %'].loc[3: 3]),
            'perf+5': float(backtest['Performance der Signale in %'].loc[4: 4]),
            'perf+6': float(backtest['Performance der Signale in %'].loc[5: 5]),
            'perf+7': float(backtest['Performance der Signale in %'].loc[6: 6]),
            'perf+8': float(backtest['Performance der Signale in %'].loc[7: 7]),
            'perf+9': float(backtest['Performance der Signale in %'].loc[8: 8]),
            'perf+10': float(backtest['Performance der Signale in %'].loc[9:9]),
            'perf+11': float(backtest['Performance der Signale in %'].loc[10: 10]),
            'perf+12': float(backtest['Performance der Signale in %'].loc[11: 11]),
            'perf+13': float(backtest['Performance der Signale in %'].loc[12: 12]),
            'perf+14': float(backtest['Performance der Signale in %'].loc[13: 13]),
            'perf+15': float(backtest['Performance der Signale in %'].loc[14: 14]),
            'perf+16': float(backtest['Performance der Signale in %'].loc[15: 15]),
            'perf+17': float(backtest['Performance der Signale in %'].loc[16: 16]),
            'perf+18': float(backtest['Performance der Signale in %'].loc[17: 17]),
            'perf+19': float(backtest['Performance der Signale in %'].loc[18: 18]),
            'perf+20': float(backtest['Performance der Signale in %'].loc[19: 19]),
            'perf+20_b&h': float(backtest['Performance buy & hold in %'].loc[19: 19]),

    }, ignore_index=True)

    df.to_csv('Mappe2.csv', sep=' ', index=False)

    print(df)


main()

