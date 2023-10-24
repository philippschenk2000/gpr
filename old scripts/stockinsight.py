import configparser
import datetime
import decimal
import indicators.bb as bb
import pandas as pd
import numpy as np
import indicators.ema200_rsi as ema200_rsi
import indicators.rsi as rsi
import indicators.rsi1882 as rsi1882
import indicators.rsi1981 as rsi1981
import indicators.rsi2080 as rsi2080
import indicators.rsi2179 as rsi2179
import indicators.rsi2278 as rsi2278
import indicators.rsi2377 as rsi2377
import indicators.rsi2476 as rsi2476
import indicators.rsi2575 as rsi2575
import indicators.rsi2674 as rsi2674
import indicators.rsi2773 as rsi2773
import indicators.rsi2872 as rsi2872
import indicators.rsi2971 as rsi2971
import indicators.rsi3070 as rsi3070
import indicators.rsi3169 as rsi3169
import indicators.rsi3268 as rsi3268
import indicators.rsi3367 as rsi3367
import indicators.rsi3466 as rsi3466
import indicators.rsi3565 as rsi3565
import indicators.rsi3664 as rsi3664
import indicators.rsi3763 as rsi3763
import indicators.rsi3862 as rsi3862
import indicators.rsi3961 as rsi3961
import indicators.rsi4060 as rsi4060
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
import backtesting.backtesting as backtestingg
import backtesting.backtesting_rsi as backtesting_rsi
import backtesting.backtesting_rsi1882 as backtesting_rsi1882
import backtesting.backtesting_rsi1981 as backtesting_rsi1981
import backtesting.backtesting_rsi2080 as backtesting_rsi2080
import backtesting.backtesting_rsi2179 as backtesting_rsi2179
import backtesting.backtesting_rsi2278 as backtesting_rsi2278
import backtesting.backtesting_rsi2377 as backtesting_rsi2377
import backtesting.backtesting_rsi2476 as backtesting_rsi2476
import backtesting.backtesting_rsi2575 as backtesting_rsi2575
import backtesting.backtesting_rsi2674 as backtesting_rsi2674
import backtesting.backtesting_rsi2773 as backtesting_rsi2773
import backtesting.backtesting_rsi2872 as backtesting_rsi2872
import backtesting.backtesting_rsi2971 as backtesting_rsi2971
import backtesting.backtesting_rsi3070 as backtesting_rsi3070
import backtesting.backtesting_rsi3169 as backtesting_rsi3169
import backtesting.backtesting_rsi3268 as backtesting_rsi3268
import backtesting.backtesting_rsi3367 as backtesting_rsi3367
import backtesting.backtesting_rsi3466 as backtesting_rsi3466
import backtesting.backtesting_rsi3565 as backtesting_rsi3565
import backtesting.backtesting_rsi3664 as backtesting_rsi3664
import backtesting.backtesting_rsi3763 as backtesting_rsi3763
import backtesting.backtesting_rsi3862 as backtesting_rsi3862
import backtesting.backtesting_rsi3961 as backtesting_rsi3961
import backtesting.backtesting_rsi4060 as backtesting_rsi4060
import backtesting.backtesting_ema as backtesting_ema
import backtesting.backtesting_fla as backtesting_fla
import backtesting.backtesting_flb as backtesting_flb
import backtesting.backtesting_flc as backtesting_flc
import backtesting.backtesting_fld as backtesting_fld
import backtesting.backtesting_cross as backtesting_cross
import backtesting.backtesting_hor4 as backtesting_hor4
import backtesting.backtesting_hor5 as backtesting_hor5
import backtesting.backtesting_hor6 as backtesting_hor6
import backtesting.backtesting_ret3 as backtesting_ret3
import backtesting.backtesting_ret1 as backtesting_ret1
import backtesting.backtesting_mret2 as backtesting_mret2
import backtesting.backtesting_mret1 as backtesting_mret1
import backtesting.backtesting_fiba as backtesting_fiba
import backtesting.backtesting_fibb as backtesting_fibb
import backtesting.backtesting_fibc as backtesting_fibc
import backtesting.backtesting_kk as backtesting_kk
import backtesting.backtesting_uk as backtesting_uk
import requests
import time
import matplotlib.pyplot as plt
import heapq


def main():
    symbol = input("Enter your symbol:")
    candlesticks = get_candlesticks(symbol)
    candlesticks.reverse()
    df1 = pd.DataFrame()
    df1 = calculate(candlesticks, df1)
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_rows', None)
    for d in df1:
        df1.append(df1)
    backtest = backtesting(df1, symbol)
    print(symbol)


def backtesting(df1, symbol):

    print(df1)

    with open(symbol + '.textmate', 'w') as file:
        file.write(str(backtestingg.df1gefiltert(df1)) + '\n')
        file.write('')
        if (backtestingg.performance_5(df1)) > -100 and (backtestingg.performance_10(df1)) > -100 and (backtestingg.performance_15(df1)) > -100 and (backtestingg.performance_20(df1))>-100:
            file.write(str(('Gesamtperformance 5D -buy', 'nach buy:', round(backtestingg.performance_5(df1)*100, 3),'%', 'Nach 10D:', round(backtestingg.performance_10(df1)*100, 3),'%', 'Nach 15D:', round(backtestingg.performance_15(df1)*100, 3),'%', 'Nach 20D:', round(backtestingg.performance_20(df1)*100, 3),'%')) + '\n')
        file.write(str(('Durchschn. Performance B&H', 'nach 5D:', round(backtestingg.performance_taeglich(df1)*100*5, 3),'%', 'Nach 10D:', round(backtestingg.performance_taeglich(df1)*100*10, 3),'%', 'Nach 15D:', round(backtestingg.performance_taeglich(df1)*100*15, 3),'%', 'Nach 20D:', round(backtestingg.performance_taeglich(df1)*100*20, 3),'%')) + '\n')
        file.write(str(('Logaritm. Performance B&H', 'nach 5D:', round(backtestingg.performance_taeglich_ln(df1)*100*5, 3),'%', 'Nach 10D:', round(backtestingg.performance_taeglich_ln(df1)*100*10, 3),'%', 'Nach 15D:', round(backtestingg.performance_taeglich_ln(df1)*100*15, 3),'%', 'Nach 20D:', round(backtestingg.performance_taeglich_ln(df1)*100*20, 3),'%')) + '\n')
        if (backtesting_rsi.performance_rsi5(df1)) > -100 and (backtesting_rsi.performance_rsi10(df1)) > -100 and (backtesting_rsi.performance_rsi15(df1)) > -100 and (backtesting_rsi.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI-buy', 'nach 5D:', round(backtesting_rsi.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_rsi1882.performance_rsi5(df1)) > -100 and (backtesting_rsi1882.performance_rsi10(df1)) > -100 and (backtesting_rsi1882.performance_rsi15(df1)) > -100 and (backtesting_rsi1882.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI1882-buy', 'nach 5D:', round(backtesting_rsi1882.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi1882.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi1882.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi1882.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_rsi1981.performance_rsi5(df1)) > -100 and (backtesting_rsi1981.performance_rsi10(df1)) > -100 and (backtesting_rsi1981.performance_rsi15(df1)) > -100 and (backtesting_rsi1981.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI1981-buy', 'nach 5D:', round(backtesting_rsi1981.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi1981.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi1981.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi1981.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_rsi2080.performance_rsi5(df1)) > -100 and (backtesting_rsi2080.performance_rsi10(df1)) > -100 and (backtesting_rsi2080.performance_rsi15(df1)) > -100 and (backtesting_rsi2080.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI2080-buy', 'nach 5D:', round(backtesting_rsi2080.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi2080.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi2080.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi2080.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_rsi2179.performance_rsi5(df1)) > -100 and (backtesting_rsi2179.performance_rsi10(df1)) > -100 and (backtesting_rsi2179.performance_rsi15(df1)) > -100 and (backtesting_rsi2179.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI2179-buy', 'nach 5D:', round(backtesting_rsi2179.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi2179.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi2179.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi2179.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_rsi2278.performance_rsi5(df1)) > -100 and (backtesting_rsi2278.performance_rsi10(df1)) > -100 and (backtesting_rsi2278.performance_rsi15(df1)) > -100 and (backtesting_rsi2278.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI2278-buy', 'nach 5D:', round(backtesting_rsi2278.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi2278.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi2278.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi2278.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_rsi2377.performance_rsi5(df1)) > -100 and (backtesting_rsi2377.performance_rsi10(df1)) > -100 and (backtesting_rsi2377.performance_rsi15(df1)) > -100 and (backtesting_rsi2377.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI2377-buy', 'nach 5D:', round(backtesting_rsi2377.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi2377.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi2377.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi2377.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_rsi2476.performance_rsi5(df1)) > -100 and (backtesting_rsi2476.performance_rsi10(df1)) > -100 and (backtesting_rsi2476.performance_rsi15(df1)) > -100 and (backtesting_rsi2476.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI2476-buy', 'nach 5D:', round(backtesting_rsi2476.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi2476.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi2476.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi2476.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_rsi2575.performance_rsi5(df1)) > -100 and (backtesting_rsi2575.performance_rsi10(df1)) > -100 and (backtesting_rsi2575.performance_rsi15(df1)) > -100 and (backtesting_rsi2575.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI2575-buy', 'nach 5D:', round(backtesting_rsi2575.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi2575.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi2575.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi2575.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_rsi2674.performance_rsi5(df1)) > -100 and (backtesting_rsi2674.performance_rsi10(df1)) > -100 and (backtesting_rsi2674.performance_rsi15(df1)) > -100 and (backtesting_rsi2674.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI2674-buy', 'nach 5D:', round(backtesting_rsi2674.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi2674.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi2674.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi2674.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_rsi2773.performance_rsi5(df1)) > -100 and (backtesting_rsi2773.performance_rsi10(df1)) > -100 and (backtesting_rsi2773.performance_rsi15(df1)) > -100 and (backtesting_rsi2773.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI2773-buy', 'nach 5D:', round(backtesting_rsi2773.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi2773.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi2773.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi2773.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_rsi2872.performance_rsi5(df1)) > -100 and (backtesting_rsi2872.performance_rsi10(df1)) > -100 and (backtesting_rsi2872.performance_rsi15(df1)) > -100 and (backtesting_rsi2872.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI2872-buy', 'nach 5D:', round(backtesting_rsi2872.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi2872.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi2872.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi2872.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_rsi2971.performance_rsi5(df1)) > -100 and (backtesting_rsi2971.performance_rsi10(df1)) > -100 and (backtesting_rsi2971.performance_rsi15(df1)) > -100 and (backtesting_rsi2971.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI2971-buy', 'nach 5D:', round(backtesting_rsi2971.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi2971.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi2971.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi2971.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_rsi3070.performance_rsi5(df1)) > -100 and (backtesting_rsi3070.performance_rsi10(df1)) > -100 and (backtesting_rsi3070.performance_rsi15(df1)) > -100 and (backtesting_rsi3070.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI3070-buy', 'nach 5D:', round(backtesting_rsi3070.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi3070.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi3070.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi3070.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_rsi3169.performance_rsi5(df1)) > -100 and (backtesting_rsi3169.performance_rsi10(df1)) > -100 and (backtesting_rsi3169.performance_rsi15(df1)) > -100 and (backtesting_rsi3169.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI3169-buy', 'nach 5D:', round(backtesting_rsi3169.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi3169.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi3169.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi3169.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_rsi3268.performance_rsi5(df1)) > -100 and (backtesting_rsi3268.performance_rsi10(df1)) > -100 and (backtesting_rsi3268.performance_rsi15(df1)) > -100 and (backtesting_rsi3268.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI3268-buy', 'nach 5D:', round(backtesting_rsi3268.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi3268.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi3268.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi3268.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_rsi3367.performance_rsi5(df1)) > -100 and (backtesting_rsi3367.performance_rsi10(df1)) > -100 and (backtesting_rsi3367.performance_rsi15(df1)) > -100 and (backtesting_rsi3367.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI3367-buy', 'nach 5D:', round(backtesting_rsi3367.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi3367.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi3367.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi3367.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_rsi3466.performance_rsi5(df1)) > -100 and (backtesting_rsi3466.performance_rsi10(df1)) > -100 and (backtesting_rsi3466.performance_rsi15(df1)) > -100 and (backtesting_rsi3466.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI3466-buy', 'nach 5D:', round(backtesting_rsi3466.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi3466.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi3466.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi3466.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_rsi3565.performance_rsi5(df1)) > -100 and (backtesting_rsi3565.performance_rsi10(df1)) > -100 and (backtesting_rsi3565.performance_rsi15(df1)) > -100 and (backtesting_rsi3565.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI3565-buy', 'nach 5D:', round(backtesting_rsi3565.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi3565.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi3565.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi3565.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_rsi3664.performance_rsi5(df1)) > -100 and (backtesting_rsi3664.performance_rsi10(df1)) > -100 and (backtesting_rsi3664.performance_rsi15(df1)) > -100 and (backtesting_rsi3664.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI3664-buy', 'nach 5D:', round(backtesting_rsi3664.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi3664.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi3664.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi3664.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_rsi3763.performance_rsi5(df1)) > -100 and (backtesting_rsi3763.performance_rsi10(df1)) > -100 and (backtesting_rsi3763.performance_rsi15(df1)) > -100 and (backtesting_rsi3763.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI3763-buy', 'nach 5D:', round(backtesting_rsi3763.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi3763.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi3763.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi3763.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_rsi3862.performance_rsi5(df1)) > -100 and (backtesting_rsi3862.performance_rsi10(df1)) > -100 and (backtesting_rsi3862.performance_rsi15(df1)) > -100 and (backtesting_rsi3862.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI3862-buy', 'nach 5D:', round(backtesting_rsi3862.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi3862.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi3862.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi3862.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_rsi3961.performance_rsi5(df1)) > -100 and (backtesting_rsi3961.performance_rsi10(df1)) > -100 and (backtesting_rsi3961.performance_rsi15(df1)) > -100 and (backtesting_rsi3961.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI3961-buy', 'nach 5D:', round(backtesting_rsi3961.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi3961.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi3961.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi3961.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_rsi4060.performance_rsi5(df1)) > -100 and (backtesting_rsi4060.performance_rsi10(df1)) > -100 and (backtesting_rsi4060.performance_rsi15(df1)) > -100 and (backtesting_rsi4060.performance_rsi20(df1))>-100:
            file.write(str(('Performance RSI4060-buy', 'nach 5D:', round(backtesting_rsi4060.performance_rsi5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_rsi4060.performance_rsi10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi4060.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi4060.performance_rsi20(df1)*100, 3),'%')) + '\n')
    '''    if (backtesting_ema.performance_ema5(df1)) > -100 and (backtesting_ema.performance_ema10(df1)) > -100 and (backtesting_ema.performance_ema15(df1)) > -100 and (backtesting_ema.performance_ema20(df1))>-100:
            file.write(str(('Performance EMA-buy', 'nach 5D:', round(backtesting_ema.performance_ema5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_ema.performance_ema10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_rsi.performance_rsi15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_rsi.performance_rsi20(df1)*100, 3),'%')) + '\n')
        if (backtesting_fla.performance_fla5(df1)) > -100 and (backtesting_fla.performance_fla10(df1)) > -100 and (backtesting_fla.performance_fla15(df1)) > -100 and (backtesting_fla.performance_fla20(df1))>-100:
            file.write(str(('Performance FLA-buy', 'nach 5D:', round(backtesting_fla.performance_fla5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_fla.performance_fla10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_fla.performance_fla15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_fla.performance_fla20(df1)*100, 3),'%')) + '\n')
        if (backtesting_flb.performance_flb5(df1))>-100 and (backtesting_flb.performance_flb10(df1))>-100 and (backtesting_flb.performance_flb15(df1))>-100 and (backtesting_flb.performance_flb20(df1))>-100:
            file.write(str(('Performance FLB-buy', 'nach 5D:', round(backtesting_flb.performance_flb5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_flb.performance_flb10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_flb.performance_flb15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_flb.performance_flb20(df1)*100, 3),'%')) + '\n')
        if (backtesting_flc.performance_flc5(df1))>-100 and (backtesting_flc.performance_flc10(df1))>-100 and (backtesting_flc.performance_flc15(df1))>-100 and (backtesting_flc.performance_flc20(df1))>-100:
            file.write(str(('Performance FLC-buy', 'nach 5D:', round(backtesting_flc.performance_flc5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_flc.performance_flc10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_flc.performance_flc15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_flc.performance_flc20(df1)*100, 3),'%')) + '\n')
        if (backtesting_fld.performance_fld5(df1))>-100 and (backtesting_fld.performance_fld10(df1))>-100 and (backtesting_fld.performance_fld15(df1))>-100 and (backtesting_fld.performance_fld20(df1))>-100:
            file.write(str(('Performance FLD-buy', 'nach 5D:', round(backtesting_fld.performance_fld5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_fld.performance_fld10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_fld.performance_fld15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_fld.performance_fld20(df1)*100, 3),'%')) + '\n')
        if (backtesting_cross.performance_cross5(df1))>-100 and (backtesting_cross.performance_cross10(df1))>-100 and (backtesting_cross.performance_cross15(df1))>-100 and (backtesting_cross.performance_cross20(df1))>-100:
            file.write(str(('Performance CROSS-buy', 'nach 5D:', round(backtesting_cross.performance_cross5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_cross.performance_cross10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_cross.performance_cross15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_cross.performance_cross20(df1)*100, 3),'%')) + '\n')
        if (backtesting_hor4.performance_ema5(df1))>-100 and (backtesting_hor4.performance_ema10(df1))>-100 and (backtesting_hor4.performance_ema15(df1))>-100 and (backtesting_hor4.performance_hor420(df1))>-100:
            file.write(str(('Performance HOR4-buy', 'nach 5D:', round(backtesting_hor4.performance_ema5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_hor4.performance_ema10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_hor4.performance_ema15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_hor4.performance_hor420(df1)*100, 3),'%')) + '\n')
        if (backtesting_hor5.performance_ema5(df1))>-100 and (backtesting_hor5.performance_ema10(df1))>-100 and (backtesting_hor5.performance_ema15(df1))>-100 and (backtesting_hor5.performance_hor520(df1))>-100:
            file.write(str(('Performance HOR5-buy', 'nach 5D:', round(backtesting_hor5.performance_ema5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_hor5.performance_ema10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_hor5.performance_ema15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_hor5.performance_hor520(df1)*100, 3),'%')) + '\n')
        if (backtesting_hor6.performance_ema5(df1))>-100 and (backtesting_hor6.performance_ema10(df1))>-100 and (backtesting_hor6.performance_ema15(df1))>-100 and (backtesting_hor6.performance_hor620(df1))>-100:
            file.write(str(('Performance HOR6-buy', 'nach 5D:', round(backtesting_hor6.performance_ema5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_hor6.performance_ema10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_hor6.performance_ema15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_hor6.performance_hor620(df1)*100, 3),'%')) + '\n')
        if (backtesting_ret3.performance_ema5(df1))>-100 and (backtesting_ret3.performance_ema10(df1))>-100 and (backtesting_ret3.performance_ema15(df1))>-100 and (backtesting_ret3.performance_ret320(df1))>-100:
            file.write(str(('Performance RET3-buy', 'nach 5D:', round(backtesting_ret3.performance_ema5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_ret3.performance_ema10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_ret3.performance_ema15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_ret3.performance_ret320(df1)*100, 3),'%')) + '\n')
        if (backtesting_ret1.performance_ema5(df1))>-100 and (backtesting_ret1.performance_ema10(df1))>-100 and (backtesting_ret1.performance_ema15(df1))>-100 and (backtesting_ret1.performance_ret120(df1))>-100:
            file.write(str(('Performance RET1-buy', 'nach 5D:', round(backtesting_ret1.performance_ema5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_ret1.performance_ema10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_ret1.performance_ema15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_ret1.performance_ret120(df1)*100, 3),'%')) + '\n')
        if (backtesting_mret2.performance_ema5(df1))>-100 and (backtesting_mret2.performance_ema10(df1))>-100 and (backtesting_mret2.performance_ema15(df1))>-100 and (backtesting_mret2.performance_mret220(df1))>-100:
            file.write(str(('Performance MRET2-buy', 'nach 5D:', round(backtesting_mret2.performance_ema5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_mret2.performance_ema10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_mret2.performance_ema15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_mret2.performance_mret220(df1)*100, 3),'%')) + '\n')
        if (backtesting_mret1.performance_ema5(df1))>-100 and (backtesting_mret1.performance_ema10(df1))>-100 and (backtesting_mret1.performance_ema15(df1))>-100 and (backtesting_mret1.performance_mret120(df1))>-100:
            file.write(str(('Performance MRET1-buy', 'nach 5D:', round(backtesting_mret1.performance_ema5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_mret1.performance_ema10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_mret1.performance_ema15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_mret1.performance_mret120(df1)*100, 3),'%')) + '\n')
        if (backtesting_fiba.performance_ema5(df1))>-100 and (backtesting_fiba.performance_ema10(df1))>-100 and (backtesting_fiba.performance_ema15(df1))>-100 and (backtesting_fiba.performance_fiba20(df1))>-100:
            file.write(str(('Performance FIBA-buy', 'nach 5D:', round(backtesting_fiba.performance_ema5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_fiba.performance_ema10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_fiba.performance_ema15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_fiba.performance_fiba20(df1)*100, 3),'%')) + '\n')
        if (backtesting_fibb.performance_ema5(df1))>-100 and (backtesting_fibb.performance_ema10(df1))>-100 and (backtesting_fibb.performance_ema15(df1))>-100 and (backtesting_fibb.performance_fibb20(df1))>-100:
            file.write(str(('Performance FIBB-buy', 'nach 5D:', round(backtesting_fibb.performance_ema5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_fibb.performance_ema10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_fibb.performance_ema15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_fibb.performance_fibb20(df1)*100, 3),'%')) + '\n')
        if (backtesting_fibc.performance_ema5(df1))>-100 and (backtesting_fibc.performance_ema10(df1))>-100 and (backtesting_fibc.performance_ema15(df1))>-100 and (backtesting_fibc.performance_fibc20(df1))>-100:
            file.write(str(('Performance FIBC-buy', 'nach 5D:', round(backtesting_fibc.performance_ema5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_fibc.performance_ema10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_fibc.performance_ema15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_fibc.performance_fibc20(df1)*100, 3),'%')) + '\n')
        if (backtesting_kk.performance_ema5(df1))>-100 and (backtesting_kk.performance_ema10(df1))>-100 and (backtesting_kk.performance_ema15(df1))>-100 and (backtesting_kk.performance_kk20(df1))>-100:
            file.write(str(('Performance KK-buy', 'nach 5D:', round(backtesting_kk.performance_ema5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_kk.performance_ema10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_kk.performance_ema15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_kk.performance_kk20(df1)*100, 3),'%')) + '\n')
        if (backtesting_uk.performance_ema5(df1))>-100 and (backtesting_uk.performance_ema10(df1))>-100 and (backtesting_uk.performance_ema15(df1))>-100 and (backtesting_uk.performance_uk20(df1))>-100:
            file.write(str(('Performance UK-buy', 'nach 5D:', round(backtesting_uk.performance_ema5(df1)*100, 3),'%', 'Nach 10D:', round(backtesting_uk.performance_ema10(df1)*100, 3),'%', 'Nach 15D:', round(backtesting_uk.performance_ema15(df1)*100, 3),'%', 'Nach 20D:', round(backtesting_uk.performance_uk20(df1)*100, 3),'%')) + '\n')'''


def get_candlesticks(symbol):
    url = "https://cloud.iexapis.com/stable/stock/" + symbol + "/chart/max"        #Euronext Brussels (BRU): -BB XETRA (ETR): -GY Euronext Paris (PAR): -FP London Stock Exchange (LON): -LN
    payload = {'token': "pk_e34bceaad39545de99147557c9c0b969", 'chartLast': 2500}
    r = requests.get(url, params=payload)
    r.raise_for_status()
    prices = r.json()
    for price in prices:
        for key in list(price):
            if key not in ["date", "high", "low", "open", "close"]:
                del price[key]
    return prices


def calculate(candlesticks, df1):
    for i in range(0, 2200):
        prices = []
        interval = candlesticks[i:i+300]
        for candlestick in interval:
            prices.append([float(candlestick['open']), float(candlestick['high']), float(candlestick['low']), float(candlestick['close'])])
        step = math.floor(((prices[-1][1] - prices[-1][2]) / 3) * 1000) / 1000
        prices[-1][3] = prices[-1][2]
        for i in range(0, 3):
            df1 = get_events(prices, interval, df1)
            prices[-1][3] += step
    return df1


def get_events(prices, interval, df1):
    '''highs_level_two = highs.get_level_two(prices)
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
    horizontal_level_six = hors.horizontal_level_six(prices, highshighs_level_six, lowslows_level_six)'''

    b = 'buy'
    s = 'sell'
    h = ''

    df = df1.append({
        'TAG': interval[-1]["date"],
        'PRICE': round(prices[-1][3], 3),
        'RSI': b if rsi.buy(prices) and FLneg.ja(prices) else (s if rsi.sell(prices) else h),
        'RSI1882': b if rsi1882.buy(prices) and FLneg.ja(prices) else (s if rsi1882.sell(prices) else h),
        'RSI1981': b if rsi1981.buy(prices) and FLneg.ja(prices) else (s if rsi1981.sell(prices) else h),
        'RSI2080': b if rsi2080.buy(prices) and FLneg.ja(prices) else (s if rsi2080.sell(prices) else h),
        'RSI2179': b if rsi2179.buy(prices) and FLneg.ja(prices) else (s if rsi2179.sell(prices) else h),
        'RSI2278': b if rsi2278.buy(prices) and FLneg.ja(prices) else (s if rsi2278.sell(prices) else h),
        'RSI2377': b if rsi2377.buy(prices) and FLneg.ja(prices) else (s if rsi2377.sell(prices) else h),
        'RSI2476': b if rsi2476.buy(prices) and FLneg.ja(prices) else (s if rsi2476.sell(prices) else h),
        'RSI2575': b if rsi2575.buy(prices) and FLneg.ja(prices) else (s if rsi2575.sell(prices) else h),
        'RSI2674': b if rsi2674.buy(prices) and FLneg.ja(prices) else (s if rsi2674.sell(prices) else h),
        'RSI2773': b if rsi2773.buy(prices) and FLneg.ja(prices) else (s if rsi2773.sell(prices) else h),
        'RSI2872': b if rsi2872.buy(prices) and FLneg.ja(prices) else (s if rsi2872.sell(prices) else h),
        'RSI2971': b if rsi2971.buy(prices) and FLneg.ja(prices) else (s if rsi2971.sell(prices) else h),
        'RSI3070': b if rsi3070.buy(prices) and FLneg.ja(prices) else (s if rsi3070.sell(prices) else h),
        'RSI3169': b if rsi3169.buy(prices) and FLneg.ja(prices) else (s if rsi3169.sell(prices) else h),
        'RSI3268': b if rsi3268.buy(prices) and FLneg.ja(prices) else (s if rsi3268.sell(prices) else h),
        'RSI3367': b if rsi3367.buy(prices) and FLneg.ja(prices) else (s if rsi3367.sell(prices) else h),
        'RSI3466': b if rsi3466.buy(prices) and FLneg.ja(prices) else (s if rsi3466.sell(prices) else h),
        'RSI3565': b if rsi3565.buy(prices) and FLneg.ja(prices) else (s if rsi3565.sell(prices) else h),
        'RSI3664': b if rsi3664.buy(prices) and FLneg.ja(prices) else (s if rsi3664.sell(prices) else h),
        'RSI3763': b if rsi3763.buy(prices) and FLneg.ja(prices) else (s if rsi3763.sell(prices) else h),
        'RSI3862': b if rsi3862.buy(prices) and FLneg.ja(prices) else (s if rsi3862.sell(prices) else h),
        'RSI3961': b if rsi3961.buy(prices) and FLneg.ja(prices) else (s if rsi3961.sell(prices) else h),
        'RSI4060': b if rsi4060.buy(prices) and FLneg.ja(prices) else (s if rsi4060.sell(prices) else h),
        #        'EMA': b if ema200_rsi.buy(prices) and FLneg.ja(prices) else (s if ema200_rsi.sell(prices) and FLpos.ja(prices) else h),
 #       'FLA': b if fla.buy(prices) and FLneg.ja(prices) else h,
  #      'FLB': b if flb.buy(prices) and FLneg.ja(prices) else h,
   #     'FLC': b if flc.buy(prices) and FLneg.ja(prices) else h,
    #    'FLD': b if fld.buy(prices) and FLneg.ja(prices) else h,
     #   'WEAKA': s if weaka.sell(prices) and FLpos.ja(prices) else h,
      #  'WEAKB': s if weakb.sell(prices) and FLpos.ja(prices) else h,
       # 'WEAKC': s if weakc.sell(prices) and FLpos.ja(prices) else h,
        #'WEAKD': s if weakd.sell(prices) and FLpos.ja(prices) else h,
#        'CROSS': b if cross.buy(prices, lines_low, horizontal_level_four) and FLneg.ja(prices) else (s if cross.sell(prices, lines_high, horizontal_level_four) and FLpos.ja(prices) else h),
 #       'HOR4': b if hors.horizontal4_buy(prices, horizontal_level_four) and FLneg.ja(prices) else (s if hors.horizontal4_sell(prices, horizontal_level_four) and FLpos.ja(prices) else h),
  #      'HOR5': b if hors.horizontal5_buy(prices, horizontal_level_five) and FLneg.ja(prices) else (s if hors.horizontal5_sell(prices, horizontal_level_five) and FLpos.ja(prices) else h),
   #     'HOR6': b if hors.horizontal6_buy(prices, horizontal_level_six) and FLneg.ja(prices) else (s if hors.horizontal6_sell(prices, horizontal_level_six) and FLpos.ja(prices) else h),
    #    'RET3': b if ret300.buy(prices, hocs_level_five) and FLneg.ja(prices) else (s if ret300.sell(prices, locs_level_five) and FLpos.ja(prices) else h),
     #   'RET1': b if ret100.buy(prices, hocs_level_four) and FLneg.ja(prices) else (s if ret100.sell(prices, locs_level_four) and FLpos.ja(prices) else h),
      #  'MRET2': b if mret200.buy(prices, highshighs_level_five) and FLneg.ja(prices) else (s if mret200.sell(prices, lowslows_level_five) and FLpos.ja(prices) else h),
       # 'MRET1': b if mret100.buy(prices, highshighs_level_four) and FLneg.ja(prices) else (s if mret100.sell(prices, lowslows_level_four) and FLpos.ja(prices) else h),
        #'FIBA': b if fiba.buy(prices, highshighs_level_five, lowslows_level_five) and FLneg.ja(prices) else h,
#        'FIBB': b if fibb.buy(prices, highshighs_level_four, lowslows_level_four) and FLneg.ja(prices) else h,
 #       'FIBC': b if fibc.buy(prices, highshighs_level_three, lowslows_level_three) and FLneg.ja(prices) else h,
  #      'KK': b if kkerze.buy(prices) and FLneg.ja(prices) else h,
   #     'UK': b if ukerze.buy(prices) and FLneg.ja(prices) else (s if ukerze.sell(prices) and FLpos.ja(prices) else h)
    }, ignore_index=True)

    return df


main()
