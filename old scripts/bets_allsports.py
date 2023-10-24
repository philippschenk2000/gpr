import datetime
import difflib
import random
import re
import smtplib
import time
import zoneinfo
from scipy import stats
from datetime import timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from threading import Thread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import requests
from deep_translator import GoogleTranslator
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import bookies.admiralbet as admiralbet
import bookies.alphawin as alphawin
import bookies.bambet as bambet
import bookies.bankonbet as bankonbet
import bookies.bcgame as bcgame
import bookies.bet3000 as bet3000
import bookies.bet90 as bet90
import bookies.betano as betano
import bookies.betathome as betathome
import bookies.betclic as betclic
import bookies.betfury as betfury
import bookies.betibet as betibet
import bookies.betmaster as betmaster
import bookies.betobet as betobet
import bookies.betplay as betplay
import bookies.betrophy as betrophy
import bookies.bets_api as bets_api
import bookies.betsamigo as betsamigo
import bookies.betstro as betstro
import bookies.betvictor as betvictor
import bookies.betway as betway
import bookies.bildbet as bildbet
import bookies.bluechip as bluechip
import bookies.bpremium as bpremium
import bookies.bwin as bwin
import bookies.campeonbet as campeonbet
import bookies.cashalot as cashalot
import bookies.casumo as casumo
import bookies.cbet as cbet
import bookies.chillybets as chillybets
import bookies.cloudbet as cloudbet
import bookies.cobrabet as cobrabet
import bookies.cricbaba as cricbaba
import bookies.dachbet as dachbet
import bookies.dafabet as dafabet
import bookies.dreambet as dreambet
import bookies.evobet as evobet
import bookies.expekt as expekt
import bookies.fdj as fdj
import bookies.fezbet as fezbet
import bookies.gamblingapes as gamblingapes
import bookies.gastonred as gastonred
import bookies.greatwin as greatwin
import bookies.happybet as happybet
import bookies.interwetten as interwetten
import bookies.ivibet as ivibet
import bookies.jets10 as jets10
import bookies.joabet as joabet
import bookies.joycasino as joycasino
import bookies.kto as kto
import bookies.ladbrokes as ladbrokes
import bookies.leonbet as leonbet
import bookies.leovegas as leovegas
import bookies.librabet as librabet
import bookies.lilibet as lilibet
import bookies.livescorebet as livescorebet
import bookies.lsbet as lsbet
import bookies.marathonbet as marathonbet
import bookies.megapari as megapari
import bookies.merkur_sports as merkur_sports
import bookies.mobilebet as mobilebet
import bookies.moonbet as moonbet
import bookies.n1bet as n1bet
import bookies.nearcasino as nearcasino
import bookies.neobet as neobet
import bookies.netbet as netbet
import bookies.nexbetsports as nexbetsports
import bookies.nextbet as nextbet
import bookies.nucleonbet as nucleonbet
import bookies.odds_api as odds_api
import bookies.oddset as oddset
import bookies.olympusbet as olympusbet
import bookies.onebet as onebet
import bookies.onexbet as onexbet
import bookies.onexbit as onexbit
import bookies.owlgames as owlgames
import bookies.paripesa as paripesa
import bookies.pinnacle as pinnacle
import bookies.playzilla as playzilla
import bookies.pmu as pmu
import bookies.pokerstars as pokerstars
import bookies.powbet as powbet
import bookies.qbet as qbet
import bookies.quickwin as quickwin
import bookies.rabona as rabona
import bookies.rajbets as rajbets
import bookies.rocketplay as rocketplay
import bookies.roobet as roobet
import bookies.sgcasino as sgcasino
import bookies.skybet as skybet
import bookies.solcasino as solcasino
import bookies.rollbit as rollbit
import bookies.sport888 as sport888
import bookies.sportaza as sportaza
import bookies.sportingbet as sportingbet
import bookies.sportwetten_de as sportwetten_de
import bookies.stake as stake
import bookies.sultanbet as sultanbet
import bookies.suprabets as suprabets
import bookies.terracasino as terracasino
import bookies.threetwored as threetwored
import bookies.tipico_api as tipico_api
import bookies.tipico as tipico
import bookies.tipwin as tipwin
import bookies.twotwobet as twotwobet
import bookies.twozerobet as twozerobet
import bookies.unibet as unibet
import bookies.vave as vave
import bookies.vbet as vbet
import bookies.virginbet as virginbet
import bookies.vistabet as vistabet
import bookies.wazamba as wazamba
import bookies.weltbet as weltbet
import bookies.wettarena as wettarena
import bookies.winamax as winamax
import bookies.winning as winning
import bookies.winz as winz
import bookies.wolfbet as wolfbet
import bookies.zebet as zebet
import bookies.zenitbet as zenitbet
import bookies.zotabet as zotabet
import bookies.mybet_api as mybet_api
import bookies.duelbits as duelbits
import bookies.threetwored2 as threetwored2
import bookies.lottoland as lottoland
import bookies.magicalvegas as magicalvegas
import bookies.tiptorro as tiptorro
import bookies.genybet as genybet
import bookies.mystake as mystake
import bookies.freshbet as freshbet
import bookies.goldenbet as goldenbet
import bookies.jackbit as jackbit
import bookies.threeonebet as threeonebet
import bookies.bet365 as bet365
import bookies.draftkings as draftkings
import bookies.fanduel as fanduel
import bookies.betmgm as betmgm
import bookies.caesars as caesars
import bookies.betuk as betuk
import bookies.pinnacle_api as pinnacle_api
import bookies.piwi247 as piwi247
import bookies.asianodds as asianodds
import bookies.ps3838 as ps3838
import bookies.parimatch as parimatch
import twitter
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def main():

    recievers = ['brv.vierer@gmail.com']#, 'mbetting19@gmail.com']#, 'baptistebachmann@googlemail.com']
    countries = ['england', 'england_cup', 'germany', 'germany_cup', 'finland', 'romania', 'czechia', 'greece', 'poland', 'scotland', 'austria', 'switzerland', 'denmark', 'italy',
                 'france', 'belgium', 'netherlands', 'spain', 'sweden', 'norway', 'argentina', 'japan', 'brazil', 'portugal', 'copalib', 'usa', 'euro', 'cl', 'el', 'ecl']
    countries = random.sample(countries, len(countries))

    tabs_count_odds = 9
    """ SOCCER """
    for j in range(0, 20):
        for c in countries:
            for i in range(0, 1):
                print(c)
                get_soccer_urls(c)
                Thread(target=get_soccer_odds2, args=(tabs_count_odds, c)).start()
                Thread(target=get_soccer_odds3, args=(tabs_count_odds, c)).start()
                Thread(target=get_soccer_odds4, args=(tabs_count_odds, c)).start()
                get_soccer_odds(tabs_count_odds, c)
                Thread(target=surebets_mail, args=(recievers, c)).start()
                """ TO DO for new country: add leagues in leagues.csv; copy & rename match&odds csv; odd in list "countries" """
        time.sleep(60*80)

# hinzufügen weiterer odds: asian handicap, handicap, evenodd, drawnobet, double chance, last goal
#https://sportsbook.copybet.com/papi/default/Popup/getPopups?data=%7B%22pageId%22%3A49%2C%22currentUrl%22%3A%22%2Fpre-match%2Fmatch%2FSoccer%2FGermany%2F541%2F23173352%22%7D&_token=
#https://www.boylesports.com/sports/football/competition/france-ligue-1?partial=true
#http://www.betcalculation.com/calculatevalue.php
#https://help.smarkets.com/hc/en-gb/articles/214554985-How-to-calculate-expected-value-in-betting
#https://www.openbet.com/partners
#contentId,3155587.1
#https://www-dazg-ssb-pr.daznbet.de/services/content/get
#https://pre-49o-sp.websbkt.com/cache/49/de/de/6669153/single-pre-event.json?hidenseek=9f33703f7ee745b20b4eef7cb651d18fc962f35e00d4
#https://altenar.com/de/clients/
#https://sportsbet.esportings.com/proxy3/api/v1.1/data/markets
#https://msports.m88.com/api/v1/M88S/data/morebet
#https://www.sbobet.com/de-DE/euro/fu%c3%9fball/europe
#https://sx.bet/soccer/champions-league/1X2/L11294385
#https://bookmaker.xyz/polygon/sports/football
#https://exchange.purebet.io/ , https://api.purebet.io/pbapi?sport=soccer
#https://www.registrierung-pin.com/
#https://sports.sportium.es/es/t/45211/La-Liga
#https://deportes.marcaapuestas.es/es/t/19160/Primera-Divisi%C3%B3n
#https://apuestas.retabet.es/deportes/futbol/laliga-s1
#https://www.marsbet.com/en/wisegaming
#https://www.10bet.com/sports/football/germany-1-bundesliga/
#https://api-web.tipwin.de/v2/100501/offer/data POST
#https://sports.betway.de/api/Events/v2/GetGroup POST
#https://cms.coral.co.uk/cms/api/bma/yc-leagues
#https://buildyourbet.prod.coral.co.uk/api/v1/events
#https://ss-aka-ori.coral.co.uk/openbet-ssviewer/Drilldown/2.31/EventToOutcomeForType/442?simpleFilter=event.eventSortCode:notIntersects:TNMT,TR01,TR02,TR03,TR04,TR05,TR06,TR07,TR08,TR09,TR10,TR11,TR12,TR13,TR14,TR15,TR16,TR17,TR18,TR19,TR20&simpleFilter=market.templateMarketName:intersects:|Match%20Betting|,|Over/Under%20Total%20Goals|,|Both%20Teams%20to%20Score|,|To%20Qualify|,|Draw%20No%20Bet|,|First-Half%20Result|,|Next%20Team%20to%20Score|,|Extra-Time%20Result|,|2Up%20-%20Instant%20Win|,|2Up%26Win%20Early%20Payout|,Match%20Betting,Over/Under%20Total%20Goals,Both%20Teams%20to%20Score,To%20Qualify,Draw%20No%20Bet,First-Half%20Result,Next%20Team%20to%20Score,Extra-Time%20Result,2Up%26Win%20Early%20Payout,2Up%20-%20Instant%20Win,Match%20Result%20and%20Both%20Teams%20To%20Score,|Match%20Result%20and%20Both%20Teams%20To%20Score|&translationLang=en&responseFormat=json&prune=event&prune=market&childCount=event
#https://ss-aka-ori.coral.co.uk/openbet-ssviewer/Drilldown/2.31/EventToOutcomeForEvent/27599104?scorecast=true&translationLang=en&responseFormat=json&referenceEachWayTerms=true (ladbrokes??)
#https://fo.wynnbet-ma-web.gansportsbook.com/s/sbgate/sports/fo-category/?country=US&language=us&layout=EUROPEAN&province&categoryId=155
#https://fo.wynnbet-ma-web.gansportsbook.com/s/sbgate/sports/fo-market/sidebets?country=US&language=us&layout=EUROPEAN&province&matchId=26982


def surebets_mail(recievers, c):

    odds_per_bookie = same_soccer_events_preprocessing(c)
    events = same_soccer_events(odds_per_bookie, c)

    surebets = get_soccer_surebets_global(events)
    surebets = surebets_preprocessing(surebets, c)
    surebets_ger = get_soccer_surebets_germany(events)
    surebets_ger = surebets_preprocessing_germany(surebets_ger, c)

    valuebets = get_soccer_valuebets(events)
    valuebets = valuebets_preprocessing(valuebets, c)

    """ NOTIFICATIONS """
    send_mails(surebets_ger, valuebets, recievers)


def get_soccer_urls(c):

    for j in range(0, 1):
        df = pd.read_csv('csv_data/date.csv')
        df = df.loc[df['country'] == c]
        if str(datetime.datetime.now(tz=zoneinfo.ZoneInfo(key='Europe/Berlin')).strftime('%Y-%m-%d')) in df['date'].tolist():
            print('URLs already saved')
            break
        else:
            if c == 'england':
                portfolio()
            date = pd.DataFrame()
            date['date'] = [str(datetime.datetime.now(tz=zoneinfo.ZoneInfo(key='Europe/Berlin')).strftime('%Y-%m-%d'))]
            date['country'] = [c]
            df1 = pd.read_csv('csv_data/date.csv')
            df1 = df1._append(date, ignore_index=True)

            tabs_count_games = 6
            df = pd.read_csv('csv_data/matches{}.csv'.format(c))
            df = df.iloc[0:0]
            df.to_csv('csv_data/matches{}.csv'.format(c), index=False)

            bookies_with_match_pages = ['merkur_sports', 'interwetten', 'caesars', 'betano', 'winamax', 'neobet', 'unibet', 'pokerstars', 'pmu', 'fdj', 'netbet', 'joabet', 'zebet', 'leonbet', 'cbet', 'dreambet', 'mobilebet', 'admiralbet', 'betmaster', 'dafabet', 'betway', 'sport888', 'happybet', 'betathome', 'tipwin', 'stake', 'chillybets', 'tipico', 'genybet', 'mystake', 'draftkings', 'fanduel', 'caesars']
            leagues1 = pd.read_csv('csv_data/leagues.csv').sample(frac=1).sort_values(by=['sports', 'country'])
            leagues = leagues1.loc[(leagues1['sports'] == 'soccer') & (leagues1['bookie'].isin(bookies_with_match_pages)) & (leagues1['country'] == c)].reset_index(drop=True)
            leagues2 = leagues1.loc[(leagues1['sports'] == 'soccer') & (~leagues1['bookie'].isin(bookies_with_match_pages)) & (leagues1['country'] == c)].reset_index(drop=True)
            leagues = leagues._append(leagues.tail(4), ignore_index=True)
            print(leagues)

            df = pd.read_csv('csv_data/matches{}.csv'.format(c))._append(leagues2, ignore_index=True)
            df.to_csv('csv_data/matches{}.csv'.format(c), index=False)

            chrome_options = Options()
            chrome_options.add_argument("--incognito")
            chrome_options.add_argument('--no-sandbox')  # Bypass OS security model
            chrome_options.add_argument('start-maximized')  #
            chrome_options.add_argument('disable-infobars')
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-dev-shm-usage")
            #chrome_options.add_argument('--user-data-dir=~/.config/google-chrome')
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--headless=new")
            driver = webdriver.Chrome(options=chrome_options, executable_path='/usr/bin/chromedriver')
            #driver = webdriver.Chrome(options=chrome_options, service=service)
            for t in range(1, tabs_count_games):
                driver.execute_script("window.open('about:blank', 'tab{}');".format(t+1))
            abbruch = int(len(leagues['sports']) - tabs_count_games/tabs_count_games+1)
            for runs in range(0, abbruch):    # anzahl der durchläufe in zweier schritten durch die leagues csv
                length = []
                for t in range(0, tabs_count_games):
                    length.append(runs * tabs_count_games + t + 1)
                this_session_leagues = leagues.loc[leagues.index.isin(length)].reset_index(drop=True)
                for l in range(0, len(this_session_leagues)):
                    driver.switch_to.window(driver.window_handles[l])
                    url = this_session_leagues['url'][l]
                    try:
                        driver.get(url)
                        print(url)
                    except: print('failed:', url)


                for l in range(0, len(this_session_leagues)):
                    driver.switch_to.window(driver.window_handles[l])
                    one_league = this_session_leagues.loc[this_session_leagues.index == l].reset_index(drop=True)
                    sports = one_league['sports'].values[0]
                    bookie = one_league['bookie'].values[0]
                    country = one_league['country'].values[0]
                    url = one_league['url'][0]
                    try:
                        if one_league['bookie'].values[0] == 'betano':
                            betano.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'interwetten':
                            interwetten.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'merkur_sports':
                            merkur_sports.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'winamax':
                            winamax.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'neobet':
                            neobet.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'unibet':
                            unibet.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'pokerstars':
                            pokerstars.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'pmu':
                            pmu.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'fdj':
                            fdj.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'joabet':
                            joabet.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'zebet':
                            zebet.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'netbet':
                            netbet.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'leonbet':
                            leonbet.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'cbet':
                            cbet.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'dreambet':
                            dreambet.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'mobilebet':
                            mobilebet.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'admiralbet':
                            admiralbet.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'betmaster':
                            betmaster.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'dafabet':
                            dafabet.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'betway':
                            betway.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'sport888':
                            sport888.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'happybet':
                            happybet.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'betathome':
                            betathome.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'tipwin':
                            tipwin.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'stake':
                            stake.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'chillybets':
                            chillybets.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'tipico':
                            tipico.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'genybet':
                            genybet.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'mystake':
                            mystake.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'draftkings':
                            draftkings.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'fanduel':
                            fanduel.soccer_urls(driver, sports, bookie, country, url, c)
                        elif one_league['bookie'].values[0] == 'caesars':
                            caesars.soccer_urls(sports, bookie, country, url, c)

                        # add more bookies with match urls
                        time.sleep(0.2)
                    except: a=1
                time.sleep(0.2)
            driver.close()
            time.sleep(0.3)
            df1.to_csv('csv_data/date.csv'.format(c), index=False)


def get_soccer_odds(tabs_count_odds, c):

    tabs_count_odds = tabs_count_odds + 1

    time.sleep(0.5)
    start_time = time.time()
    df = pd.read_csv('csv_data/odds{}.csv'.format(c), delimiter=',')
    df = df.iloc[0:0]
    df.to_csv('csv_data/odds{}.csv'.format(c), index=False, sep=',')
    bookmaker = ['winamax', 'betway', 'joabet', 'sport888', 'happybet', 'interwetten', 'dreambet', 'neobet', 'wettarena']

    matches = pd.read_csv('csv_data/matches{}.csv'.format(c)).sample(frac=1).sort_values(by=['sports', 'country'])
    matches = matches.drop_duplicates()
    matches = matches.loc[(matches['sports'] == 'soccer') & (matches['country'] == c) & (matches['bookie'].isin(bookmaker))].reset_index(drop=True)
    matches['game_url'] = matches['game_url'].fillna(matches['url'])
    print(matches.head(8))

    chrome_options = Options()
    chrome_options.add_argument("--incognito")
    chrome_options.add_argument("--disable-site-isolation-trials")
    chrome_options.add_argument('--no-sandbox')  # Bypass OS security model
    chrome_options.add_argument('start-maximized')  #
    chrome_options.add_argument('disable-infobars')
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--enable-javascript")

    driver = webdriver.Chrome(options=chrome_options, executable_path='/usr/bin/chromedriver')
    for t in range(1, tabs_count_odds):
        driver.execute_script("window.open('about:blank', 'tab{}');".format(t + 1))
    abbruch = int((len(matches)-tabs_count_odds) / tabs_count_odds + 1)

    match_odds = pd.DataFrame()
    for runs in range(0, abbruch):    # anzahl der durchläufe in zweier schritten durch die matches csv
        length = []
        for t in range(0, tabs_count_odds):
            length.append(runs * tabs_count_odds + t)
        this_session_leagues = matches.loc[matches.index.isin(length)].reset_index(drop=True)
        for l in range(0, len(this_session_leagues)):
            driver.switch_to.window(driver.window_handles[l])
            try:
                url = this_session_leagues['game_url'][l]
                driver.set_page_load_timeout(10)
                driver.get(url)
            except: a=1

        for l in range(0, len(this_session_leagues)):
            driver.switch_to.window(driver.window_handles[l])
            one_league = this_session_leagues.loc[this_session_leagues.index == l].reset_index(drop=True)
            sports = one_league['sports'].values[0]
            bookie = one_league['bookie'].values[0]
            country = one_league['country'].values[0]
            url = one_league['game_url'][0]
            sleeping = 0.0
            try:
                print(url)
                total = ''
                if one_league['bookie'].values[0] == 'winamax':
                    total = winamax.soccer_odds(driver, sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'betway':
                    total = betway.soccer_odds(driver, sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'wettarena':
                    total = wettarena.soccer_odds(driver, sports, bookie, country, url)
                elif one_league['bookie'].values[0] == 'joabet':
                    total = joabet.soccer_odds(driver, sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'sport888':
                    total = sport888.soccer_odds(driver, sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'happybet':
                    total = happybet.soccer_odds(driver, sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'interwetten':
                    total = interwetten.soccer_odds(driver, sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'dreambet':
                    dreambet_total = dreambet.soccer_odds(driver, sports, bookie, country, url)
                    onebet_total = onebet.soccer_odds(dreambet_total)
                    betobet_total = betobet.soccer_odds(dreambet_total)
                    weltbet_total = weltbet.soccer_odds(dreambet_total)
                    olympusbet_total = olympusbet.soccer_odds(dreambet_total)
                    betrophy_total = betrophy.soccer_odds(dreambet_total)
                    dachbet_total = dachbet.soccer_odds(dreambet_total)
                    cashalot_total = cashalot.soccer_odds(dreambet_total)
                    total = dreambet_total._append(onebet_total, ignore_index=True)
                    total = total._append(betobet_total, ignore_index=True)
                    total = total._append(weltbet_total, ignore_index=True)
                    total = total._append(olympusbet_total, ignore_index=True)
                    total = total._append(betrophy_total, ignore_index=True)
                    total = total._append(dachbet_total, ignore_index=True)
                    total = total._append(cashalot_total, ignore_index=True)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'neobet':
                    total = neobet.soccer_odds(driver, sports, bookie, country, url)
                    time.sleep(sleeping)


                match_odds = match_odds._append(total, ignore_index=True)
            except:
                print('failed to retrieve or not a game: ' + url)
    match_odds2 = match_odds.head(1)
    match_odds2['bookie'] = ['fertiq']
    match_odds = match_odds._append(match_odds2, ignore_index=True)
    while 'fertik' not in list(pd.read_csv('csv_data/odds{}.csv'.format(c))['bookie']): time.sleep(3)
    while 'fertic' not in list(pd.read_csv('csv_data/odds{}.csv'.format(c))['bookie']): time.sleep(3)
    while 'fertig' not in list(pd.read_csv('csv_data/odds{}.csv'.format(c))['bookie']): time.sleep(3)
    odds = pd.read_csv('csv_data/odds{}.csv'.format(c), delimiter=',')._append(match_odds, ignore_index=True)
    odds.to_csv('csv_data/odds{}.csv'.format(c), index=False, sep=',')
    driver.close()
    print('First part went thru')
    print('Duration: ', str(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))


def get_soccer_odds2(tabs_count_odds, c):

    tabs_count_odds = tabs_count_odds + 1
    time.sleep(0.5)
    bookmaker = ['unibet', 'pokerstars', 'netbet', 'draftkings', 'caesars', 'fanduel', 'leonbet', 'sportwetten_de', 'ladbrokes', 'bet3000', 'betathome', 'mybet', 'duelbits', 'bcgame', 'genybet']
    matches = pd.read_csv('csv_data/matches{}.csv'.format(c)).sample(frac=1).sort_values(by=['sports', 'country'])
    matches = matches.drop_duplicates()
    matches = matches.loc[(matches['sports'] == 'soccer') & (matches['country'] == c) & (matches['bookie'].isin(bookmaker))].reset_index(drop=True)
    matches['game_url'] = matches['game_url'].fillna(matches['url'])
    print(matches.head(8))

    abbruch = int((len(matches)-tabs_count_odds) / tabs_count_odds + 1)

    match_odds = pd.DataFrame()
    for runs in range(0, abbruch):    # anzahl der durchläufe in zweier schritten durch die matches csv
        length = []
        for t in range(0, tabs_count_odds):
            length.append(runs * tabs_count_odds + t)
        this_session_leagues = matches.loc[matches.index.isin(length)].reset_index(drop=True)

        for l in range(0, len(this_session_leagues)):
            one_league = this_session_leagues.loc[this_session_leagues.index == l].reset_index(drop=True)
            sports = one_league['sports'].values[0]
            bookie = one_league['bookie'].values[0]
            country = one_league['country'].values[0]
            url = one_league['game_url'][0]
            sleeping = 0.4
            try:
                print(url)
                total = ''
                if one_league['bookie'].values[0] == 'unibet':
                    total = unibet.soccer_odds(sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'pokerstars':
                    total = pokerstars.soccer_odds(sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'draftkings':
                    total = draftkings.soccer_odds(sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'caesars':
                    total = caesars.soccer_odds(sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'fanduel':
                    total = fanduel.soccer_odds(sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'netbet':
                    total = netbet.soccer_odds(sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'leonbet':
                    total = leonbet.soccer_odds(sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'sportwetten_de':
                    total = sportwetten_de.soccer_odds(sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'ladbrokes':
                    ladbrokes_total = ladbrokes.soccer_odds(sports, bookie, country, url)
                    bwin_total = bwin.soccer_odds(ladbrokes_total)
                    sportingbet_total = sportingbet.soccer_odds(ladbrokes_total)
                    oddset_total = oddset.soccer_odds(ladbrokes_total)
                    bpremium_total = bpremium.soccer_odds(ladbrokes_total)
                    vistabet_total = vistabet.soccer_odds(ladbrokes_total)
                    betmgm_total = betmgm.soccer_odds(ladbrokes_total)
                    total = ladbrokes_total._append(bwin_total, ignore_index=True)
                    total = total._append(sportingbet_total, ignore_index=True)
                    total = total._append(oddset_total, ignore_index=True)
                    total = total._append(bpremium_total, ignore_index=True)
                    total = total._append(vistabet_total, ignore_index=True)
                    total = total._append(betmgm_total, ignore_index=True)
                    time.sleep(sleeping)
                #elif one_league['bookie'].values[0] == 'betmaster':
                #    total = betmaster.soccer_odds(sports, bookie, country, url)
                #    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'bet3000':
                    total = bet3000.soccer_odds(sports, bookie, country, url)
                elif one_league['bookie'].values[0] == 'betathome':
                    total = betathome.soccer_odds(sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'bcgame':
                    bcgame_total = bcgame.soccer_odds(sports, bookie, country, url)
                    total = bcgame_total
                    try:
                        roobet_total = roobet.soccer_odds(bcgame_total)
                        solcasino_total = solcasino.soccer_odds(bcgame_total)
                        rollbit_total = rollbit.soccer_odds(bcgame_total)
                        nearcasino_total = nearcasino.soccer_odds(bcgame_total)
                        gamblingapes_total = gamblingapes.soccer_odds(bcgame_total)
                        joycasino_total = joycasino.soccer_odds(bcgame_total)
                        moonbet_total = moonbet.soccer_odds(bcgame_total)
                        bluechip_total = bluechip.soccer_odds(bcgame_total)
                        rajbets_total = rajbets.soccer_odds(bcgame_total)
                        betfury_total = betfury.soccer_odds(bcgame_total)
                        owlgames_total = owlgames.soccer_odds(bcgame_total)
                        terracasino_total = terracasino.soccer_odds(bcgame_total)
                        total = bcgame_total._append(roobet_total, ignore_index=True)
                        total = total._append(solcasino_total, ignore_index=True)
                        total = total._append(rollbit_total, ignore_index=True)
                        total = total._append(nearcasino_total, ignore_index=True)
                        total = total._append(gamblingapes_total, ignore_index=True)
                        total = total._append(joycasino_total, ignore_index=True)
                        total = total._append(moonbet_total, ignore_index=True)
                        total = total._append(bluechip_total, ignore_index=True)
                        total = total._append(rajbets_total, ignore_index=True)
                        total = total._append(betfury_total, ignore_index=True)
                        total = total._append(owlgames_total, ignore_index=True)
                        total = total._append(terracasino_total, ignore_index=True)
                    except: a=1
                elif one_league['bookie'].values[0] == 'mybet':
                    mybet_total = mybet_api.soccer_odds(sports, bookie, country, url)
                    total = mybet_total
                    try:
                        expekt_total = expekt.soccer_odds(mybet_total)
                        total = mybet_total._append(expekt_total, ignore_index=True)
                        casumo_total = casumo.soccer_odds(mybet_total)
                        total = total._append(casumo_total, ignore_index=True)
                        leovegas_total = leovegas.soccer_odds(mybet_total)
                        total = total._append(leovegas_total, ignore_index=True)
                        threetwored2_total = threetwored2.soccer_odds(mybet_total)
                        total = total._append(threetwored2_total, ignore_index=True)
                        betplay_total = betplay.soccer_odds(mybet_total)
                        total = total._append(betplay_total, ignore_index=True)
                        betuk_total = betuk.soccer_odds(mybet_total)
                        total = total._append(betuk_total, ignore_index=True)
                    except: a=1
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'duelbits':
                    total = duelbits.soccer_odds(sports, bookie, country, url)
                elif one_league['bookie'].values[0] == 'genybet':
                    total = genybet.soccer_odds(sports, bookie, country, url)
                    time.sleep(sleeping)


                # weitere hinzufügen, doch erst in bet_scraper 1 ausprobieren
                match_odds = match_odds._append(total, ignore_index=True)
            except:
                print('failed to retrieve or not a game: ' + url)
        time.sleep(0.0)
    match_odds2 = match_odds.head(1).reset_index(drop=True)
    match_odds2['bookie'] = ['fertig']
    match_odds = match_odds._append(match_odds2, ignore_index=True)
    print('Second part went thru')
    while 'fertik' not in list(pd.read_csv('csv_data/odds{}.csv'.format(c))['bookie']): time.sleep(3)
    while 'fertic' not in list(pd.read_csv('csv_data/odds{}.csv'.format(c))['bookie']): time.sleep(3)
    odds = pd.read_csv('csv_data/odds{}.csv'.format(c), delimiter=',')._append(match_odds, ignore_index=True)
    odds.to_csv('csv_data/odds{}.csv'.format(c), index=False, sep=',')


def get_soccer_odds3(tabs_count_odds, c):

    tabs_count_odds = tabs_count_odds + 1
    time.sleep(0.5)
    bookmaker = ['betano', 'tiptorro', 'lsbet', 'stake', 'chillybets', 'dafabet', 'cloudbet', 'pinnacle', 'tipico', 'fdj', 'bet365', 'betvictor', '22bet', 'virginbet', 'betclic', 'vave', 'mystake', 'n1bet']
    matches = pd.read_csv('csv_data/matches{}.csv'.format(c)).sample(frac=1).sort_values(by=['sports', 'country'])
    matches = matches.drop_duplicates()
    matches = matches.loc[(matches['sports'] == 'soccer') & (matches['country'] == c) & (matches['bookie'].isin(bookmaker))].reset_index(drop=True)
    matches['game_url'] = matches['game_url'].fillna(matches['url'])
    print(matches.head(8))

    abbruch = int((len(matches)-tabs_count_odds) / tabs_count_odds + 1)

    match_odds = pd.DataFrame()
    for runs in range(0, abbruch):    # anzahl der durchläufe in zweier schritten durch die matches csv
        length = []
        for t in range(0, tabs_count_odds):
            length.append(runs * tabs_count_odds + t)
        this_session_leagues = matches.loc[matches.index.isin(length)].reset_index(drop=True)

        for l in range(0, len(this_session_leagues)):
            one_league = this_session_leagues.loc[this_session_leagues.index == l].reset_index(drop=True)
            sports = one_league['sports'].values[0]
            bookie = one_league['bookie'].values[0]
            country = one_league['country'].values[0]
            url = one_league['game_url'][0]
            sleeping = 0.4
            try:
                print(url)
                total = ''
                if one_league['bookie'].values[0] == 'betano':
                    total = betano.soccer_odds(sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'tiptorro':
                    total = tiptorro.soccer_odds(sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'lsbet':
                    total = lsbet.soccer_odds(sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'stake':
                    total = stake.soccer_odds(sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'chillybets':
                    chillybets_total = chillybets.soccer_odds(sports, bookie, country, url)
                    onexbet_total = onexbet.soccer_odds(chillybets_total)
                    gastonred_total = gastonred.soccer_odds(chillybets_total)
                    librabet_total = librabet.soccer_odds(chillybets_total)
                    campeonbet_total = campeonbet.soccer_odds(chillybets_total)
                    alphawin_total = alphawin.soccer_odds(chillybets_total)
                    kto_total = kto.soccer_odds(chillybets_total)
                    fezbet_total = fezbet.soccer_odds(chillybets_total)
                    powbet_total = powbet.soccer_odds(chillybets_total)
                    sportaza_total = sportaza.soccer_odds(chillybets_total)
                    evobet_total = evobet.soccer_odds(chillybets_total)
                    quickwin_total = quickwin.soccer_odds(chillybets_total)
                    bankonbet_total = bankonbet.soccer_odds(chillybets_total)
                    sgcasino_total = sgcasino.soccer_odds(chillybets_total)
                    betsamigo_total = betsamigo.soccer_odds(chillybets_total)
                    greatwin_total = greatwin.soccer_odds(chillybets_total)
                    playzilla_total = playzilla.soccer_odds(chillybets_total)
                    nucleonbet_total = nucleonbet.soccer_odds(chillybets_total)
                    rabona_total = rabona.soccer_odds(chillybets_total)
                    betstro_total = betstro.soccer_odds(chillybets_total)
                    wazamba_total = wazamba.soccer_odds(chillybets_total)
                    lottoland_total = lottoland.soccer_odds(chillybets_total)
                    magicalvegas_total = magicalvegas.soccer_odds(chillybets_total)
                    total = chillybets_total._append(onexbet_total, ignore_index=True)
                    total = total._append(gastonred_total, ignore_index=True)
                    total = total._append(librabet_total, ignore_index=True)
                    total = total._append(campeonbet_total, ignore_index=True)
                    total = total._append(alphawin_total, ignore_index=True)
                    total = total._append(kto_total, ignore_index=True)
                    total = total._append(fezbet_total, ignore_index=True)
                    total = total._append(powbet_total, ignore_index=True)
                    total = total._append(sportaza_total, ignore_index=True)
                    total = total._append(evobet_total, ignore_index=True)
                    total = total._append(quickwin_total, ignore_index=True)
                    total = total._append(bankonbet_total, ignore_index=True)
                    total = total._append(sgcasino_total, ignore_index=True)
                    total = total._append(betsamigo_total, ignore_index=True)
                    total = total._append(greatwin_total, ignore_index=True)
                    total = total._append(playzilla_total, ignore_index=True)
                    total = total._append(nucleonbet_total, ignore_index=True)
                    total = total._append(rabona_total, ignore_index=True)
                    total = total._append(betstro_total, ignore_index=True)
                    total = total._append(wazamba_total, ignore_index=True)
                    total = total._append(lottoland_total, ignore_index=True)
                    total = total._append(magicalvegas_total, ignore_index=True)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'dafabet':
                    total = dafabet.soccer_odds(sports, bookie, country, url)
                    try:
                        nextbet_total = nextbet.soccer_odds(total)
                        total = total._append(nextbet_total, ignore_index=True)
                    except: a=1
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'cloudbet':
                    total = cloudbet.soccer_odds(sports, bookie, country, url)
                elif one_league['bookie'].values[0] == 'pinnacle':
                    pinnacle_total = pinnacle_api.soccer_odds(sports, bookie, country, url)
                    piwi247_total = piwi247.soccer_odds(pinnacle_total)
                    ps3838_total = ps3838.soccer_odds(pinnacle_total)
                    asianodds_total = asianodds.soccer_odds(pinnacle_total)
                    total = pinnacle_total._append(piwi247_total, ignore_index=True)
                    total = total._append(ps3838_total, ignore_index=True)
                    total = total._append(asianodds_total, ignore_index=True)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'tipico':
                    total = tipico.soccer_odds(sports, bookie, country, url)
                elif one_league['bookie'].values[0] == 'fdj':
                    total = fdj.soccer_odds(sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'bet365':
                    total = bet365.soccer_odds(sports, bookie, country, url)
                elif one_league['bookie'].values[0] == 'betvictor':
                    time.sleep(sleeping)
                    betvictor_total = betvictor.soccer_odds(sports, bookie, country, url)
                    bildbet_total = bildbet.soccer_odds(betvictor_total)
                    parimatch_total = parimatch.soccer_odds(betvictor_total)
                    total = betvictor_total._append(bildbet_total, ignore_index=True)
                    total = total._append(parimatch_total, ignore_index=True)
                elif one_league['bookie'].values[0] == '22bet':
                    twotwobet_total = twotwobet.soccer_odds(sports, bookie, country, url)
                    paripesa_total = paripesa.soccer_odds(twotwobet_total)
                    total = twotwobet_total._append(paripesa_total, ignore_index=True)
                    megapari_total = megapari.soccer_odds(twotwobet_total)
                    total = total._append(megapari_total, ignore_index=True)
                    onexbit_total = onexbit.soccer_odds(twotwobet_total)
                    total = total._append(onexbit_total, ignore_index=True)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'virginbet':
                    virginbet_total = virginbet.soccer_odds(sports, bookie, country, url)
                    skybet_total = skybet.soccer_odds(virginbet_total)
                    livescorebet_total = livescorebet.soccer_odds(virginbet_total)
                    total = virginbet_total._append(skybet_total, ignore_index=True)
                    total = total._append(livescorebet_total, ignore_index=True)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'betclic':
                    total = betclic.soccer_odds(sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'vave':
                    vave_total = vave.soccer_odds(sports, bookie, country, url)
                    twozerobet_total = twozerobet.soccer_odds(vave_total)
                    ivibet_total = ivibet.soccer_odds(vave_total)
                    total = vave_total._append(twozerobet_total, ignore_index=True)
                    total = total._append(ivibet_total, ignore_index=True)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'mystake':
                    mystake_total = mystake.soccer_odds(sports, bookie, country, url)
                    freshbet_total = freshbet.soccer_odds(mystake_total)
                    goldenbet_total = goldenbet.soccer_odds(mystake_total)
                    jackbit_total = jackbit.soccer_odds(mystake_total)
                    threeonebet_total = threeonebet.soccer_odds(mystake_total)
                    total = mystake_total._append(freshbet_total, ignore_index=True)
                    total = total._append(goldenbet_total, ignore_index=True)
                    total = total._append(jackbit_total, ignore_index=True)
                    total = total._append(threeonebet_total, ignore_index=True)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'n1bet':
                    n1bet_total = n1bet.soccer_odds(sports, bookie, country, url)
                    bambet_total = bambet.soccer_odds(n1bet_total)
                    cobrabet_total = cobrabet.soccer_odds(n1bet_total)
                    rocketplay_total = rocketplay.soccer_odds(n1bet_total)
                    qbet_total = qbet.soccer_odds(n1bet_total)
                    winz_total = winz.soccer_odds(n1bet_total)
                    betibet_total = betibet.soccer_odds(n1bet_total)
                    winning_total = winning.soccer_odds(n1bet_total)
                    zotabet_total = zotabet.soccer_odds(n1bet_total)
                    total = n1bet_total._append(bambet_total, ignore_index=True)
                    total = total._append(cobrabet_total, ignore_index=True)
                    total = total._append(rocketplay_total, ignore_index=True)
                    total = total._append(qbet_total, ignore_index=True)
                    total = total._append(winz_total, ignore_index=True)
                    total = total._append(betibet_total, ignore_index=True)
                    total = total._append(winning_total, ignore_index=True)
                    total = total._append(zotabet_total, ignore_index=True)
                    time.sleep(sleeping)

                match_odds = match_odds._append(total, ignore_index=True)
            except:
                print('failed to retrieve or not a game: ' + url)
        time.sleep(0.0)
    match_odds2 = match_odds.head(1)
    match_odds2['bookie'] = ['fertik']
    match_odds = match_odds._append(match_odds2, ignore_index=True)
    print('Third part went thru')
    while 'fertic' not in list(pd.read_csv('csv_data/odds{}.csv'.format(c))['bookie']): time.sleep(3)
    odds = pd.read_csv('csv_data/odds{}.csv'.format(c), delimiter=',')._append(match_odds, ignore_index=True)
    odds.to_csv('csv_data/odds{}.csv'.format(c), index=False, sep=',')


def get_soccer_odds4(tabs_count_odds, c):

    tabs_count_odds = tabs_count_odds + 1
    time.sleep(0.5)
    bookmaker = ['pmu', 'tipwin', 'merkur_sports', 'wolfbet', 'cbet', 'admiralbet', 'mobilebet', 'zebet']
    matches = pd.read_csv('csv_data/matches{}.csv'.format(c)).sample(frac=1).sort_values(by=['sports', 'country'])
    matches = matches.drop_duplicates()
    matches = matches.loc[(matches['sports'] == 'soccer') & (matches['country'] == c) & (matches['bookie'].isin(bookmaker))].reset_index(drop=True)
    matches['game_url'] = matches['game_url'].fillna(matches['url'])
    print(matches.head(8))

    chrome_options = Options()
    chrome_options.add_argument("--incognito")
    chrome_options.add_argument('--no-sandbox')  # Bypass OS security model
    chrome_options.add_argument('start-maximized')  #
    chrome_options.add_argument('disable-infobars')
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--headless")

    driver = webdriver.Chrome(options=chrome_options, executable_path='/usr/bin/chromedriver')
    for t in range(1, tabs_count_odds):
        driver.execute_script("window.open('about:blank', 'tab{}');".format(t + 1))
    abbruch = int((len(matches)-tabs_count_odds) / tabs_count_odds + 1)

    match_odds = pd.DataFrame()
    for runs in range(0, abbruch):    # anzahl der durchläufe in zweier schritten durch die matches csv
        length = []
        for t in range(0, tabs_count_odds):
            length.append(runs * tabs_count_odds + t)
        this_session_leagues = matches.loc[matches.index.isin(length)].reset_index(drop=True)
        for l in range(0, len(this_session_leagues)):
            driver.switch_to.window(driver.window_handles[l])
            try:
                url = this_session_leagues['game_url'][l]
                driver.get(url)
            except: a = 1

        for l in range(0, len(this_session_leagues)):
            driver.switch_to.window(driver.window_handles[l])
            one_league = this_session_leagues.loc[this_session_leagues.index == l].reset_index(drop=True)
            sports = one_league['sports'].values[0]
            bookie = one_league['bookie'].values[0]
            country = one_league['country'].values[0]
            url = one_league['game_url'][0]
            sleeping = 0.0
            try:
                print(url)
                total = ''
                if one_league['bookie'].values[0] == 'pmu':
                    total = pmu.soccer_odds(driver, sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'tipwin':
                    total = tipwin.soccer_odds(driver, sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'merkur_sports':
                    total = merkur_sports.soccer_odds(driver, sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'zebet':
                    total = zebet.soccer_odds(driver, sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'wolfbet':
                    total = wolfbet.soccer_odds(driver, sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'cbet':
                    total = cbet.soccer_odds(driver, sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'admiralbet':
                    total = admiralbet.soccer_odds(driver, sports, bookie, country, url)
                    time.sleep(sleeping)
                elif one_league['bookie'].values[0] == 'mobilebet':
                    mobilebet_total = mobilebet.soccer_odds(driver, sports, bookie, country, url)
                    jets10_total = jets10.soccer_odds(mobilebet_total)
                    sultanbet_total = sultanbet.soccer_odds(mobilebet_total)
                    lilibet_total = lilibet.soccer_odds(mobilebet_total)
                    cricbaba_total = cricbaba.soccer_odds(mobilebet_total)
                    total = mobilebet_total._append(jets10_total, ignore_index=True)
                    total = total._append(sultanbet_total, ignore_index=True)
                    total = total._append(lilibet_total, ignore_index=True)
                    total = total._append(cricbaba_total, ignore_index=True)
                    time.sleep(sleeping)


                # weitere hinzufügen, doch erst in bet_scraper 1 ausprobieren
                match_odds = match_odds._append(total, ignore_index=True)
            except:
                print('failed to retrieve or not a game: ' + url)
        time.sleep(0.1)
    match_odds2 = match_odds.head(1)
    match_odds2['bookie'] = ['fertic']
    match_odds = match_odds._append(match_odds2, ignore_index=True)
    odds = pd.read_csv('csv_data/odds{}.csv'.format(c), delimiter=',')._append(match_odds, ignore_index=True)
    odds.to_csv('csv_data/odds{}.csv'.format(c), index=False, sep=',')
    driver.close()
    print('Fourth part went thru')


def same_soccer_events_preprocessing(c):

    odds_per_bookie = pd.read_csv('csv_data/odds{}.csv'.format(c), delimiter=',')
    odds_per_bookie = odds_per_bookie.drop_duplicates()

    if len(odds_per_bookie) > 2:
        #print('These are the different bookmaker in the system: ' + '\n' + str(odds_per_bookie['bookie'].value_counts()))
        odds_per_bookie['home_team'] = odds_per_bookie['home_team'].replace('VfL BOCHUM', 'VfL Bochum')
        odds_per_bookie['match'] = odds_per_bookie['home_team'] + ' vs ' + odds_per_bookie['away_team']
        odds_per_bookie['home'] = odds_per_bookie['home'].replace('<htm', None)
        odds_per_bookie = odds_per_bookie.loc[odds_per_bookie['bookie'] != 'fertig'].reset_index(drop=True)
        odds_per_bookie = odds_per_bookie.loc[odds_per_bookie['bookie'] != 'fertik'].reset_index(drop=True)
        odds_per_bookie = odds_per_bookie.loc[odds_per_bookie['bookie'] != 'fertic'].reset_index(drop=True)
        odds_per_bookie = odds_per_bookie.loc[odds_per_bookie['bookie'] != 'fertiq'].reset_index(drop=True)
        odds_per_bookie['home'] = odds_per_bookie['home'].replace('<html cl', None)
        odds_per_bookie['url'] = odds_per_bookie['url'].fillna(odds_per_bookie['bookie'])


        for ii in range(0, 1):
            odds_per_bookie['home'] = odds_per_bookie['home'].astype(str)
            odds_per_bookie['home'] = odds_per_bookie['home'].str.replace(r'[!@#$()"%^*?:}/\{;<>~` abcdefghijklmnopqrstuvwxyzéABCDEFGHIJKLMNOPQRSTUVWXYZ-]', '', regex=True)
            odds_per_bookie['home'] = odds_per_bookie['home'].str.replace(',', '.', regex=False)
            odds_per_bookie['home'] = odds_per_bookie['home'].replace('', None)

            columns_to_modify = ['home', 'draw', 'away', 'b_score_y', 'b_score_n', 'o45', 'u45', 'o35', 'u35', 'o25', 'u25', 'o15', 'u15', 'o05', 'u05',
                                 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1', 'hand02_X', 'hand02_2', 'hand01_1', 'hand01_X', 'hand01_2',
                                 'hand10_1', 'hand10_X', 'hand10_2', 'hand20_1', 'hand20_X', 'hand20_2', 'hand30_1', 'hand30_X', 'hand30_2',
                                 'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1', 'first_h_X', 'first_h_2']
            for column in columns_to_modify:
                odds_per_bookie[column] = odds_per_bookie[column].astype(str)
                odds_per_bookie[column] = odds_per_bookie[column].str.replace(',', '.')
                odds_per_bookie[column] = odds_per_bookie[column].str.replace('"', '')
                odds_per_bookie[column] = odds_per_bookie[column].str.replace(r'[!@#$()"%^*<>?:}\/{;~"` abcdefghijklmnopqrstuvwxyzéABCDEFGHIJKLMNOPQRSTUVWXYZ-]', '', regex=True)
                odds_per_bookie[column] = odds_per_bookie[column].str.replace('\\', '')
                odds_per_bookie[column] = odds_per_bookie[column].str.replace(',', '.')
                odds_per_bookie[column] = odds_per_bookie[column].str.replace('[', '')
                odds_per_bookie[column] = odds_per_bookie[column].str.replace(']', '')
                odds_per_bookie[column] = odds_per_bookie[column].replace('', None)
                odds_per_bookie[column] = odds_per_bookie[column].str.replace('1.00', '')
                '''
                newo = []
                for o in range(0, len(odds_per_bookie[column])):
                    oo = odds_per_bookie[column][o]
                    if len(str(oo).split('.')) > 1:
                        newo.append(None)
                    else:
                        newo.append(oo)
                odds_per_bookie[column] = newo
                '''


            x = []
            for i in odds_per_bookie['time']:
                if str(i) == 'nan':
                    x.append(None)
                else:
                    x.append(re.sub(r'[!@#$()"%^*?}/{;~`ABCDEFéGHIJKLMNOPQRSTUVWXYZ ]', '', str(i)))
            odds_per_bookie['time'] = x

            odds_per_bookie['country'] = odds_per_bookie['country'].astype(str)
            odds_per_bookie['country'] = odds_per_bookie['country'].str.replace(r'[0-9]', '', regex=True)
            odds_per_bookie['country'] = odds_per_bookie['country'].replace('uefa', 'euro')
            odds_per_bookie['match'] = odds_per_bookie['match'].astype(str)
            odds_per_bookie['match'] = odds_per_bookie['match'].str.replace(' fc', '').replace('fc', '').replace(' FC', '').replace(
                'FC', '').replace(' u19', ' j-team ').replace(' u23', ' j-team ').replace(' u ', ' j-team ').replace(
                ' U19', ' j-team ').replace(' U20', ' j-team ').replace(' U21', ' j-team ').replace(' U23', ' j-team ').str.lower().replace(r'[0-9]', '', regex=True)


        texts_df = pd.DataFrame()
        texts_df['match2'] = list(set((odds_per_bookie['match'])))
        try:
            translated = GoogleTranslator(source='auto', target='en').translate_batch(texts_df['match2'].tolist())
            texts_df['match'] = translated
        except:
            texts_df['match'] = texts_df['match2']
        odds_per_bookie = pd.merge(odds_per_bookie, texts_df, on='match', how='left')
        del odds_per_bookie['match2']

        url_mapping = {
            'williamhill': 'https://sports.williamhill.com/betting/en-gb',
            'marathon bet': 'https://www.marathonbet.com/en/betting/Football',
            'boylesports': 'https://www.boylesports.com/sports/football'
        }
        odds_per_bookie['url'] = odds_per_bookie['url'].apply(lambda x: url_mapping.get(x, x))

        x = []
        for i in odds_per_bookie['match']:
            i = i.replace('\u00f6', 'ö').replace('\u00d6', 'ö').replace('\u00c4', 'ä').replace('\u00e4', 'ä')
            i = i.replace('\u00dc', 'ü').replace('\u00fc', 'ü').replace('\u00df', 'ß').replace('\u00e9', 'e')
            i = i.replace('\u00ee', 'i').replace('%C3%BC', 'ü').replace('%C3%B6', 'ö').replace('%C3%A4', 'ä').replace('\\', '')
            i = i.replace('-\\uc', 'e').replace('hove albion;', '').replace('Hove Albion;', '').replace(' &amp;', 'and')
            i = i.replace('club', '').replace(' Y ', ' ').replace(' y ', ' ').replace(' de ', ' ').replace(' pr ', ' ')
            i = i.replace(' mg', ' ').replace(' rs', ' ').replace(' fb', ' ').replace(' fr ', ' ').replace(' rj ', ' ')
            i = i.replace(' sp ', ' ').replace('sc  ', ' ').replace(' fk ', ' ').replace(' FK ', ' ').replace(' bk ', ' ')
            i = i.replace(' BK ', ' ').replace('ifk', '').replace('IFK', ' ').replace('FF', ' ').replace(' ff ', ' ')
            i = i.replace(' u19', ' j-team ').replace(' u20', ' j-team ').replace(' u21', ' j-team ').replace(' u23', ' j-team ')
            i = i.replace(' u ', ' j-team ').replace(' ii', ' j-team ').replace(' 2 ', ' j-team ').replace(' U19', ' j-team ')
            i = i.replace(' U20', ' j-team ').replace(' U21', ' j-team ').replace(' U23', ' j-team ').replace('cologne', 'koln')
            i = i.replace('Cologne', 'koln').replace('-', ' ').replace(' ca ', ' ').replace('ca r', 'r').replace(' women', ' w')
            i = i.replace(' sc', ' ').replace('SC', ' ').replace(' dc', ' ').replace(' DC', 'r')
            i = i.replace('austria klagenfurt', 'klagenfurt').replace('austria lustenau', 'lustenau').replace('Wycombe Wanderers', 'Wycombe')
            i = i.replace('calcio', '').replace('Calcio', '')
            if 'j-team' in i:
                x.append('a')
            else:
                x.append(re.sub(r'[!@#$()"%^*?:/.;~`0-9]', '', i.lower()))
        odds_per_bookie['match'] = x
        odds_per_bookie = odds_per_bookie.loc[odds_per_bookie['match'] != 'a'].reset_index(drop=True)
        odds_per_bookie = odds_per_bookie.loc[odds_per_bookie['match'] != 'nan'].reset_index(drop=True)
        odds_per_bookie = odds_per_bookie.loc[odds_per_bookie['match'] != 'home vs away'].reset_index(drop=True)

        return odds_per_bookie


def same_soccer_events(odds_per_bookie, c):

    odds_per_bookie = odds_per_bookie.drop_duplicates()
    unique_sports = odds_per_bookie['sport'].unique()
    unique_countries = odds_per_bookie['country'].unique()
    odds_per_bookie.sort_values(by='scraped_date', ascending=False)
    odds_per_bookie.to_csv('csv_data/odds4{}.csv'.format(c), index=False)
    print(sorted(odds_per_bookie['bookie'].unique().tolist()))
    print(len(odds_per_bookie['bookie'].unique().tolist()))

    df_sorted = pd.DataFrame()
    for i in unique_sports:
        for j in unique_countries:
            df = odds_per_bookie.loc[(odds_per_bookie['sport'] == i) & (odds_per_bookie['country'] == j)].reset_index(drop=True)
            print(len(df))
            # Den bookie nehmen der am längsten ist?
            df = df.drop_duplicates(subset=['bookie', 'match']).reset_index(drop=True)

            bet365 = len(df.loc[df['bookie'] == 'bet365'])
            cloudbet = len(df.loc[df['bookie'] == 'cloudbet'])
            pinnacle = len(df.loc[df['bookie'] == 'pinnacle'])
            if max(bet365, cloudbet, pinnacle) == bet365:
                perfect_names = df.loc[df['bookie'] == 'bet365'].reset_index(drop=True)
            elif max(bet365, cloudbet, pinnacle) == pinnacle:
                perfect_names = df.loc[df['bookie'] == 'pinnacle'].reset_index(drop=True)
            elif max(bet365, cloudbet, pinnacle) == cloudbet:
                perfect_names = df.loc[df['bookie'] == 'cloudbet'].reset_index(drop=True)
            else:
                break
            print(perfect_names)


            for games in range(0, len(perfect_names['match'])):
                matchdate = perfect_names['date'][games]
                matchtime = perfect_names['time'][games]
                bookie = perfect_names['bookie'][games]
                target_datetime = datetime.datetime.strptime(matchdate + ' ' + matchtime, "%Y-%m-%d %H:%M:%S")
                if bookie != 'cloudbet':
                    target_datetime += timedelta(hours=2)
                time_difference = target_datetime - datetime.datetime.now()
                print(target_datetime, time_difference)
                if time_difference.days != 0 or time_difference.seconds / 60 / 60 > 2:
                #if time_difference.seconds / 60 / 60 > 1 and time_difference.days !=:
                        game1 = perfect_names['match'][games]
                        if 'vs' in str(game1):
                            home_team = perfect_names['match'][games].split('vs')[0]
                            away_team = perfect_names['match'][games].split('vs')[1]

                            belonging_ids = [game1]
                            for bb in range(0, len(df['match'])):
                                b = df['match'][bb]
                                seq = difflib.SequenceMatcher(None, game1, b).ratio() * 100
                                if seq > 56:
                                    belonging_ids.append(bb)

                            belonging_ids = list(dict.fromkeys(belonging_ids))
                            df_matched = df.loc[df.index.isin(belonging_ids)].reset_index(drop=True)
                            df_matched['id'] = str(j) + str(games)

                            belonging_ids = []
                            for anzahl in range(0, len(df_matched['match'])):
                                if 'vs' in str(df_matched['match'][anzahl]):
                                    unkown_home_team = df_matched['match'][anzahl].split('vs')[0]
                                    unkown_away_team = df_matched['match'][anzahl].split('vs')[1]

                                    seq_homes = difflib.SequenceMatcher(None, home_team, unkown_home_team).ratio() * 100
                                    hardcountries = ['argentina', 'brazil', 'world', 'greece', 'romania', 'euro']
                                    semicountries = ['usa', 'denmark', 'norway', 'czechia', 'spain', 'italy']
                                    if j in hardcountries:
                                        if seq_homes > 70:
                                            seq_aways = difflib.SequenceMatcher(None, away_team, unkown_away_team).ratio() * 100
                                            if seq_aways > 69:
                                                belonging_ids.append(anzahl)
                                    elif j in semicountries:
                                        if seq_homes > 66:
                                            seq_aways = difflib.SequenceMatcher(None, away_team, unkown_away_team).ratio() * 100
                                            if seq_aways > 65:
                                                belonging_ids.append(anzahl)
                                    else:
                                        if seq_homes > 64:
                                            seq_aways = difflib.SequenceMatcher(None, away_team, unkown_away_team).ratio() * 100
                                            if seq_aways > 63:
                                                belonging_ids.append(anzahl)

                            df_one_game = df_matched.loc[df_matched.index.isin(belonging_ids)].reset_index(drop=True)
                            df_one_game['match'] = str(game1)
                            dates_list = df_one_game['date'].unique().tolist()
                            print(dates_list)
                            long_dates = [date for date in dates_list if len(str(date)) > 5]
                            if len(long_dates) > 1:
                                df_one_game['date'].fillna(matchdate, inplace=True)
                                df_one_game['time'].fillna(matchtime, inplace=True)
                                print('here we have to loc!')
                                df_one_game = df_one_game.loc[df_one_game['date'] == matchdate].reset_index(drop=True)
                            df_one_game['date'].fillna(matchdate, inplace=True)
                            df_one_game['time'].fillna(matchtime, inplace=True)
                            print('Anzahl von: ', game1, len(df_one_game))
                            # print(round(np.std(df_one_game['home'].astype(float).dropna().tolist()) * 10000, 2)
                            df_sorted = df_sorted._append(df_one_game, ignore_index=True)

    numeric_columns = ['home', 'draw', 'away', 'b_score_y', 'b_score_n', 'o45', 'u45', 'o35', 'u35', 'o25',
                       'u25', 'o15', 'u15', 'o05', 'u05', 'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1',
                       'first_h_X', 'first_h_2', 'hand30_1', 'hand30_X', 'hand30_2', 'hand20_1', 'hand20_X', 'hand20_2',
                       'hand10_1', 'hand10_X', 'hand10_2']
    df_sorted[numeric_columns] = df_sorted[numeric_columns].apply(pd.to_numeric, downcast='float')
    df_sorted.drop_duplicates(subset=['bookie', 'match'], keep='last', inplace=True)
    df_sorted.to_csv('csv_data/matched_odds4{}.csv'.format(c), index=False)

    reciprocal_columns = ['p_home', 'p_draw', 'p_away', 'p_b_score_y', 'p_b_score_n', 'p_o45', 'p_u45', 'p_o35', 'p_u35', 'p_o25',
                          'p_u25', 'p_o15', 'p_u15', 'p_o05', 'p_u05', 'p_first_g_1', 'p_first_g_X', 'p_first_g_2', 'p_first_h_1',
                          'p_first_h_X', 'p_first_h_2', 'p_hand30_1', 'p_hand30_X', 'p_hand30_2', 'p_hand20_1', 'p_hand20_X', 'p_hand20_2',
                          'p_hand10_1', 'p_hand10_X', 'p_hand10_2']
    df_sorted[reciprocal_columns] = 1 / df_sorted[numeric_columns].astype(float)

    return df_sorted


def get_soccer_surebets_global(events):

    df_surebets = pd.DataFrame()
    for k in events['id'].unique():
        for i in range(0, 1):
            this_bet = events.loc[events['id'] == k]
            if len(this_bet) > 1:

                # HOME, DRAW, AWAY
                for ii in range(0, 1):
                    home = this_bet.loc[this_bet['home'] == max(this_bet['home'])].reset_index(drop=True)
                    draw = this_bet.loc[this_bet['draw'] == max(this_bet['draw'])].reset_index(drop=True)
                    away = this_bet.loc[this_bet['away'] == max(this_bet['away'])].reset_index(drop=True)

                    columns_to_reset = ['o45', 'u45', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15', 'u05', 'o05',
                                        'b_score_y', 'b_score_n', 'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1',
                                        'first_h_X', 'first_h_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1', 'hand02_X',
                                        'hand02_2', 'hand01_1', 'hand01_X', 'hand01_2', 'hand10_1', 'hand10_X', 'hand10_2',
                                        'hand20_1', 'hand20_X', 'hand20_2', 'hand30_1', 'hand30_X', 'hand30_2']
                    # Reset columns to None
                    home[columns_to_reset] = None
                    home['draw'] = None
                    home['away'] = None
                    draw[columns_to_reset] = None
                    draw['home'] = None
                    draw['away'] = None
                    away[columns_to_reset] = None
                    away['home'] = None
                    away['draw'] = None
                    if len(home) > 0 and len(draw) > 0 and len(away) > 0:
                        addition = home['p_home'][0] + draw['p_draw'][0] + away['p_away'][0]
                        home['eur'] = ((100 / addition) / home['home'][0]).round(1).astype(float)
                        draw['eur'] = ((100 / addition) / draw['draw'][0]).round(1).astype(float)
                        away['eur'] = ((100 / addition) / away['away'][0]).round(1).astype(float)

                        if home['p_home'][0] + draw['p_draw'][0] + away['p_away'][0] < 1.03:
                            df_surebet = pd.concat([home, draw, away], ignore_index=True)
                            df_surebet['profit_in_%'] = round(
                                100 / (home['p_home'][0] + draw['p_draw'][0] + away['p_away'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen

                # b_score_y, b_score_n
                for ii in range(0, 1):
                    b_score_y = this_bet.loc[this_bet['b_score_y'] == max(this_bet['b_score_y'])].reset_index(drop=True)
                    b_score_n = this_bet.loc[this_bet['b_score_n'] == max(this_bet['b_score_n'])].reset_index(drop=True)
                    columns_to_reset = ['home', 'draw', 'away', 'o45', 'u45', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15', 'u05', 'o05',
                                        'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1',
                                        'first_h_X', 'first_h_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1', 'hand02_X',
                                        'hand02_2', 'hand01_1', 'hand01_X', 'hand01_2', 'hand10_1', 'hand10_X', 'hand10_2',
                                        'hand20_1', 'hand20_X', 'hand20_2', 'hand30_1', 'hand30_X', 'hand30_2']
                    # Reset columns to None
                    b_score_y[columns_to_reset] = None
                    b_score_y['b_score_n'] = None
                    b_score_n[columns_to_reset] = None
                    b_score_n['b_score_y'] = None
                    if len(b_score_n) > 0 and len(b_score_y) > 0:
                        addition = b_score_y['p_b_score_y'][0] + b_score_n['p_b_score_n'][0]
                        b_score_y['eur'] = ((100 / addition) / b_score_y['b_score_y'][0]).round(1).astype(float)
                        b_score_n['eur'] = ((100 / addition) / b_score_n['b_score_n'][0]).round(1).astype(float)
                        if b_score_y['p_b_score_y'][0] + b_score_n['p_b_score_n'][0] < 1.03:
                            df_surebet = b_score_y._append(b_score_n, ignore_index=True)
                            df_surebet['profit_in_%'] = round(100 / (b_score_y['p_b_score_y'][0] + b_score_n['p_b_score_n'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen

                # o45, u45
                for ii in range(0, 1):
                    o45 = this_bet.loc[this_bet['o45'] == max(this_bet['o45'])].reset_index(drop=True)
                    u45 = this_bet.loc[this_bet['u45'] == max(this_bet['u45'])].reset_index(drop=True)
                    columns_to_reset = ['home', 'draw', 'away', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15',
                                        'u05', 'o05', 'b_score_y', 'b_score_n',
                                        'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1',
                                        'first_h_X', 'first_h_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1',
                                        'hand02_X',
                                        'hand02_2', 'hand01_1', 'hand01_X', 'hand01_2', 'hand10_1', 'hand10_X',
                                        'hand10_2',
                                        'hand20_1', 'hand20_X', 'hand20_2', 'hand30_1', 'hand30_X', 'hand30_2']
                    # Reset columns to None
                    o45[columns_to_reset] = None
                    o45['u45'] = None
                    u45[columns_to_reset] = None
                    u45['o45'] = None
                    if len(o45) > 0 and len(u45) > 0:
                        addition = o45['p_o45'][0] + u45['p_u45'][0]
                        o45['eur'] = ((100 / addition) / o45['o45'][0]).round(1).astype(float)
                        u45['eur'] = ((100 / addition) / u45['u45'][0]).round(1).astype(float)
                        if o45['p_o45'][0] + u45['p_u45'][0] < 1.03:
                            df_surebet = o45._append(u45, ignore_index=True)
                            df_surebet['profit_in_%'] = round(100 / (o45['p_o45'][0] + u45['p_u45'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen

                # o35, u35
                for ii in range(0, 1):
                    o45 = this_bet.loc[this_bet['o35'] == max(this_bet['o35'])].reset_index(drop=True)
                    u45 = this_bet.loc[this_bet['u35'] == max(this_bet['u35'])].reset_index(drop=True)
                    columns_to_reset = ['home', 'draw', 'away', 'o45', 'u45', 'u25', 'o25', 'o15', 'u15',
                                        'u05', 'o05', 'b_score_y', 'b_score_n',
                                        'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1',
                                        'first_h_X', 'first_h_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1',
                                        'hand02_X',
                                        'hand02_2', 'hand01_1', 'hand01_X', 'hand01_2', 'hand10_1', 'hand10_X',
                                        'hand10_2',
                                        'hand20_1', 'hand20_X', 'hand20_2', 'hand30_1', 'hand30_X', 'hand30_2']
                    # Reset columns to None
                    o45[columns_to_reset] = None
                    o45['u35'] = None
                    u45[columns_to_reset] = None
                    u45['o35'] = None
                    if len(o45) > 0 and len(u45) > 0:
                        addition = o45['p_o35'][0] + u45['p_u35'][0]
                        o45['eur'] = ((100 / addition) / o45['o35'][0]).round(1).astype(float)
                        u45['eur'] = ((100 / addition) / u45['u35'][0]).round(1).astype(float)
                        if o45['p_o35'][0] + u45['p_u35'][0] < 1.03:
                            df_surebet = o45._append(u45, ignore_index=True)
                            df_surebet['profit_in_%'] = round(100 / (o45['p_o35'][0] + u45['p_u35'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen

                # o25, u25
                for ii in range(0, 1):
                    o45 = this_bet.loc[this_bet['o25'] == max(this_bet['o25'])].reset_index(drop=True)
                    u45 = this_bet.loc[this_bet['u25'] == max(this_bet['u25'])].reset_index(drop=True)
                    columns_to_reset = ['home', 'draw', 'away', 'o45', 'u45', 'u35', 'o35', 'o15', 'u15',
                                        'u05', 'o05', 'b_score_y', 'b_score_n',
                                        'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1',
                                        'first_h_X', 'first_h_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1',
                                        'hand02_X',
                                        'hand02_2', 'hand01_1', 'hand01_X', 'hand01_2', 'hand10_1', 'hand10_X',
                                        'hand10_2',
                                        'hand20_1', 'hand20_X', 'hand20_2', 'hand30_1', 'hand30_X', 'hand30_2']
                    # Reset columns to None
                    o45[columns_to_reset] = None
                    o45['u25'] = None
                    u45[columns_to_reset] = None
                    u45['o25'] = None
                    if len(o45) > 0 and len(u45) > 0:
                        addition = o45['p_o25'][0] + u45['p_u25'][0]
                        o45['eur'] = ((100 / addition) / o45['o25'][0]).round(1).astype(float)
                        u45['eur'] = ((100 / addition) / u45['u25'][0]).round(1).astype(float)
                        if o45['p_o25'][0] + u45['p_u25'][0] < 1.03:
                            df_surebet = o45._append(u45, ignore_index=True)
                            df_surebet['profit_in_%'] = round(100 / (o45['p_o25'][0] + u45['p_u25'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen

                # o15, u15
                for ii in range(0, 1):
                    o45 = this_bet.loc[this_bet['o15'] == max(this_bet['o15'])].reset_index(drop=True)
                    u45 = this_bet.loc[this_bet['u15'] == max(this_bet['u15'])].reset_index(drop=True)
                    columns_to_reset = ['home', 'draw', 'away', 'o35', 'u35', 'u25', 'o25', 'o45', 'u45',
                                        'u05', 'o05', 'b_score_y', 'b_score_n',
                                        'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1',
                                        'first_h_X', 'first_h_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1',
                                        'hand02_X',
                                        'hand02_2', 'hand01_1', 'hand01_X', 'hand01_2', 'hand10_1', 'hand10_X',
                                        'hand10_2',
                                        'hand20_1', 'hand20_X', 'hand20_2', 'hand30_1', 'hand30_X', 'hand30_2']
                    # Reset columns to None
                    o45[columns_to_reset] = None
                    o45['u15'] = None
                    u45[columns_to_reset] = None
                    u45['o15'] = None
                    if len(o45) > 0 and len(u45) > 0:
                        addition = o45['p_o15'][0] + u45['p_u15'][0]
                        o45['eur'] = ((100 / addition) / o45['o15'][0]).round(1).astype(float)
                        u45['eur'] = ((100 / addition) / u45['u15'][0]).round(1).astype(float)
                        if o45['p_o15'][0] + u45['p_u15'][0] < 1.03:
                            df_surebet = o45._append(u45, ignore_index=True)
                            df_surebet['profit_in_%'] = round(100 / (o45['p_o15'][0] + u45['p_u15'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen

                # o05, u05
                for ii in range(0, 1):
                    o45 = this_bet.loc[this_bet['o05'] == max(this_bet['o05'])].reset_index(drop=True)
                    u45 = this_bet.loc[this_bet['u05'] == max(this_bet['u05'])].reset_index(drop=True)
                    columns_to_reset = ['home', 'draw', 'away', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15',
                                        'u45', 'o45', 'b_score_y', 'b_score_n',
                                        'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1',
                                        'first_h_X', 'first_h_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1',
                                        'hand02_X',
                                        'hand02_2', 'hand01_1', 'hand01_X', 'hand01_2', 'hand10_1', 'hand10_X',
                                        'hand10_2',
                                        'hand20_1', 'hand20_X', 'hand20_2', 'hand30_1', 'hand30_X', 'hand30_2']
                    # Reset columns to None
                    o45[columns_to_reset] = None
                    o45['u05'] = None
                    u45[columns_to_reset] = None
                    u45['o05'] = None
                    if len(o45) > 0 and len(u45) > 0:
                        addition = o45['p_o05'][0] + u45['p_u05'][0]
                        o45['eur'] = ((100 / addition) / o45['o05'][0]).round(1).astype(float)
                        u45['eur'] = ((100 / addition) / u45['u05'][0]).round(1).astype(float)
                        if o45['p_o05'][0] + u45['p_u05'][0] < 1.03:
                            df_surebet = o45._append(u45, ignore_index=True)
                            df_surebet['profit_in_%'] = round(100 / (o45['p_o05'][0] + u45['p_u05'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen

                # First Goal: HOME, DRAW, AWAY
                for ii in range(0, 1):
                    home = this_bet.loc[this_bet['first_g_1'] == max(this_bet['first_g_1'])].reset_index(drop=True)
                    draw = this_bet.loc[this_bet['first_g_X'] == max(this_bet['first_g_X'])].reset_index(drop=True)
                    away = this_bet.loc[this_bet['first_g_2'] == max(this_bet['first_g_2'])].reset_index(drop=True)
                    columns_to_reset = ['home', 'draw', 'away', 'o45', 'u45', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15', 'u05', 'o05',
                                        'b_score_y', 'b_score_n', 'first_h_1',
                                        'first_h_X', 'first_h_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1',
                                        'hand02_X',
                                        'hand02_2', 'hand01_1', 'hand01_X', 'hand01_2', 'hand10_1', 'hand10_X',
                                        'hand10_2',
                                        'hand20_1', 'hand20_X', 'hand20_2', 'hand30_1', 'hand30_X', 'hand30_2']
                    # Reset columns to None
                    home[columns_to_reset] = None
                    home['first_g_X'] = None
                    home['first_g_2'] = None
                    draw[columns_to_reset] = None
                    draw['first_g_1'] = None
                    draw['first_g_2'] = None
                    away[columns_to_reset] = None
                    away['first_g_1'] = None
                    away['first_g_X'] = None
                    if len(home) > 0 and len(draw) > 0 and len(away) > 0:
                        addition = home['p_first_g_1'][0] + draw['p_first_g_X'][0] + away['p_first_g_2'][0]
                        home['eur'] = ((100 / addition) / home['first_g_1'][0]).round(1).astype(float)
                        draw['eur'] = ((100 / addition) / draw['first_g_X'][0]).round(1).astype(float)
                        away['eur'] = ((100 / addition) / away['first_g_2'][0]).round(1).astype(float)
                        if home['p_first_g_1'][0] + draw['p_first_g_X'][0] + away['p_first_g_2'][0] < 1.03:
                            df_surebet = home._append(draw, ignore_index=True)
                            df_surebet = df_surebet._append(away, ignore_index=True)
                            df_surebet['profit_in_%'] = round(
                                100 / (home['p_first_g_1'][0] + draw['p_first_g_X'][0] + away['p_first_g_2'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen

                # First half: HOME, DRAW, AWAY
                for ii in range(0, 1):
                    home = this_bet.loc[this_bet['first_h_1'] == max(this_bet['first_h_1'])].reset_index(drop=True)
                    draw = this_bet.loc[this_bet['first_h_X'] == max(this_bet['first_h_X'])].reset_index(drop=True)
                    away = this_bet.loc[this_bet['first_h_2'] == max(this_bet['first_h_2'])].reset_index(drop=True)
                    columns_to_reset = ['home', 'draw', 'away', 'o45', 'u45', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15', 'u05', 'o05',
                                        'b_score_y', 'b_score_n', 'first_g_1', 'first_g_X', 'first_g_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1',
                                        'hand02_X',
                                        'hand02_2', 'hand01_1', 'hand01_X', 'hand01_2', 'hand10_1', 'hand10_X',
                                        'hand10_2',
                                        'hand20_1', 'hand20_X', 'hand20_2', 'hand30_1', 'hand30_X', 'hand30_2']
                    # Reset columns to None
                    home[columns_to_reset] = None
                    home['first_h_X'] = None
                    home['first_h_2'] = None
                    draw[columns_to_reset] = None
                    draw['first_h_1'] = None
                    draw['first_h_2'] = None
                    away[columns_to_reset] = None
                    away['first_h_1'] = None
                    away['first_h_X'] = None
                    if len(home) > 0 and len(draw) > 0 and len(away) > 0:
                        addition = home['p_first_h_1'][0] + draw['p_first_h_X'][0] + away['p_first_h_2'][0]
                        home['eur'] = ((100 / addition) / home['first_h_1'][0]).round(1).astype(float)
                        draw['eur'] = ((100 / addition) / draw['first_h_X'][0]).round(1).astype(float)
                        away['eur'] = ((100 / addition) / away['first_h_2'][0]).round(1).astype(float)
                        if home['p_first_h_1'][0] + draw['p_first_h_X'][0] + away['p_first_h_2'][0] < 1.03:
                            df_surebet = home._append(draw, ignore_index=True)
                            df_surebet = df_surebet._append(away, ignore_index=True)
                            df_surebet['profit_in_%'] = round(
                                100 / (home['p_first_h_1'][0] + draw['p_first_h_X'][0] + away['p_first_h_2'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen


                # HANDICAP 3:0
                for ii in range(0, 1):
                    home = this_bet.loc[this_bet['hand30_1'] == max(this_bet['hand30_1'])].reset_index(drop=True)
                    draw = this_bet.loc[this_bet['hand30_X'] == max(this_bet['hand30_X'])].reset_index(drop=True)
                    away = this_bet.loc[this_bet['hand30_2'] == max(this_bet['hand30_2'])].reset_index(drop=True)

                    columns_to_reset = ['home', 'draw', 'away', 'o45', 'u45', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15', 'u05', 'o05',
                                        'b_score_y', 'b_score_n', 'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1',
                                        'first_h_X', 'first_h_2', 'hand02_1', 'hand03_1', 'hand03_X', 'hand03_2',
                                        'hand02_X',
                                        'hand02_2', 'hand01_1', 'hand01_X', 'hand01_2', 'hand10_1', 'hand10_X',
                                        'hand10_2',
                                        'hand20_1', 'hand20_X', 'hand20_2']
                    # Reset columns to None
                    home[columns_to_reset] = None
                    home['hand30_X'] = None
                    home['hand30_2'] = None
                    draw[columns_to_reset] = None
                    draw['hand30_1'] = None
                    draw['hand30_2'] = None
                    away[columns_to_reset] = None
                    away['hand30_1'] = None
                    away['hand30_X'] = None
                    if len(home) > 0 and len(draw) > 0 and len(away) > 0:
                        addition = home['p_hand30_1'][0] + draw['p_hand30_X'][0] + away['p_hand30_2'][0]
                        home['eur'] = ((100 / addition) / home['hand30_1'][0]).round(1).astype(float)
                        draw['eur'] = ((100 / addition) / draw['hand30_X'][0]).round(1).astype(float)
                        away['eur'] = ((100 / addition) / away['hand30_2'][0]).round(1).astype(float)

                        if home['p_hand30_1'][0] + draw['p_hand30_X'][0] + away['p_hand30_2'][0] < 1.03:
                            df_surebet = pd.concat([home, draw, away], ignore_index=True)
                            df_surebet['profit_in_%'] = round(
                                100 / (home['p_hand30_1'][0] + draw['p_hand30_X'][0] + away['p_hand30_2'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen

                # HANDICAP 2:0
                for ii in range(0, 1):
                    home = this_bet.loc[this_bet['hand20_1'] == max(this_bet['hand20_1'])].reset_index(drop=True)
                    draw = this_bet.loc[this_bet['hand20_X'] == max(this_bet['hand20_X'])].reset_index(drop=True)
                    away = this_bet.loc[this_bet['hand20_2'] == max(this_bet['hand20_2'])].reset_index(drop=True)

                    columns_to_reset = ['home', 'draw', 'away', 'o45', 'u45', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15', 'u05', 'o05',
                                        'b_score_y', 'b_score_n', 'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1',
                                        'first_h_X', 'first_h_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1', 'hand02_X', 'hand02_2',
                                        'hand01_1', 'hand01_X', 'hand01_2', 'hand10_1', 'hand10_X',
                                        'hand10_2', 'hand30_1', 'hand30_X', 'hand30_2']
                    # Reset columns to None
                    home[columns_to_reset] = None
                    home['hand20_X'] = None
                    home['hand20_2'] = None
                    draw[columns_to_reset] = None
                    draw['hand20_1'] = None
                    draw['hand20_2'] = None
                    away[columns_to_reset] = None
                    away['hand20_1'] = None
                    away['hand20_X'] = None
                    if len(home) > 0 and len(draw) > 0 and len(away) > 0:
                        addition = home['p_hand20_1'][0] + draw['p_hand20_X'][0] + away['p_hand20_2'][0]
                        home['eur'] = ((100 / addition) / home['hand20_1'][0]).round(1).astype(float)
                        draw['eur'] = ((100 / addition) / draw['hand20_X'][0]).round(1).astype(float)
                        away['eur'] = ((100 / addition) / away['hand20_2'][0]).round(1).astype(float)

                        if home['p_hand20_1'][0] + draw['p_hand20_X'][0] + away['p_hand20_2'][0] < 1.03:
                            df_surebet = pd.concat([home, draw, away], ignore_index=True)
                            df_surebet['profit_in_%'] = round(
                                100 / (home['p_hand20_1'][0] + draw['p_hand20_X'][0] + away['p_hand20_2'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen

                # HANDICAP 1:0
                for ii in range(0, 1):
                    home = this_bet.loc[this_bet['hand10_1'] == max(this_bet['hand10_1'])].reset_index(drop=True)
                    draw = this_bet.loc[this_bet['hand10_X'] == max(this_bet['hand10_X'])].reset_index(drop=True)
                    away = this_bet.loc[this_bet['hand10_2'] == max(this_bet['hand10_2'])].reset_index(drop=True)

                    columns_to_reset = ['home', 'draw', 'away', 'o45', 'u45', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15', 'u05', 'o05',
                                        'b_score_y', 'b_score_n', 'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1',
                                        'first_h_X', 'first_h_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1', 'hand02_X', 'hand02_2',
                                        'hand01_1', 'hand01_X', 'hand01_2',
                                        'hand20_1', 'hand20_X', 'hand20_2', 'hand30_1', 'hand30_X', 'hand30_2']
                    # Reset columns to None
                    home[columns_to_reset] = None
                    home['hand10_X'] = None
                    home['hand10_2'] = None
                    draw[columns_to_reset] = None
                    draw['hand10_1'] = None
                    draw['hand10_2'] = None
                    away[columns_to_reset] = None
                    away['hand10_1'] = None
                    away['hand10_X'] = None
                    if len(home) > 0 and len(draw) > 0 and len(away) > 0:
                        addition = home['p_hand10_1'][0] + draw['p_hand10_X'][0] + away['p_hand10_2'][0]
                        home['eur'] = ((100 / addition) / home['hand10_1'][0]).round(1).astype(float)
                        draw['eur'] = ((100 / addition) / draw['hand10_X'][0]).round(1).astype(float)
                        away['eur'] = ((100 / addition) / away['hand10_2'][0]).round(1).astype(float)

                        if home['p_hand10_1'][0] + draw['p_hand10_X'][0] + away['p_hand10_2'][0] < 1.03:
                            df_surebet = pd.concat([home, draw, away], ignore_index=True)
                            df_surebet['profit_in_%'] = round(
                                100 / (home['p_hand10_1'][0] + draw['p_hand10_X'][0] + away['p_hand10_2'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen


    if len(df_surebets) > 0:
        df_surebets = df_surebets.sort_values(by=['profit_in_%', 'id'], ascending=False)
        df_surebets = df_surebets.drop_duplicates(subset=['bookie', 'id', 'eur'], keep='last')
        df_surebets = df_surebets.loc[(df_surebets['profit_in_%'] > 0.1) & (df_surebets['profit_in_%'] < 20)].reset_index(drop=True)
        df_surebets['link'] = df_surebets['url']
        del df_surebets['country'], df_surebets['p_home'], df_surebets['p_draw'], df_surebets['p_away'], df_surebets['p_b_score_y'], df_surebets['p_b_score_n'], df_surebets['p_o45'], df_surebets['p_u45'], df_surebets['p_o35'], df_surebets['p_u35']
        del df_surebets['url'], df_surebets['p_o25'], df_surebets['p_u25'], df_surebets['p_o15'], df_surebets['p_u15'], df_surebets['p_o05'], df_surebets['p_u05']
        del df_surebets['p_first_g_1'], df_surebets['p_first_g_X'], df_surebets['p_first_g_2'], df_surebets['p_first_h_1'], df_surebets['p_first_h_X'], df_surebets['p_first_h_2']
        del df_surebets['p_hand30_1'], df_surebets['p_hand30_X'], df_surebets['p_hand30_2'], df_surebets['p_hand20_1'], df_surebets['p_hand20_X'], df_surebets['p_hand20_2'], df_surebets['p_hand10_1'], df_surebets['p_hand10_X'], df_surebets['p_hand10_2']
        for x in df_surebets['id'].unique():
            surebetsx = df_surebets.loc[df_surebets['id'] == x]
            non_deletable_cols = []
            for col in surebetsx.columns:
                uniq = surebetsx[col].unique().tolist()
                uu = []
                for u in uniq:
                    if u != '':
                        uu.append(u)
                if len(uu) > 0:
                    non_deletable_cols.append(col)
            surebetsx = surebetsx[non_deletable_cols]
            print(surebetsx)
            # if len(surebetsx.columns) > 20:
            #df_surebets = df_surebets.loc[df_surebets['id'] != k].reset_index(drop=True)

    return df_surebets


def surebets_preprocessing(surebets, c):

    bookie_url_mapping = {
        'marathon bet': 'https://www.marathonbet.com/en/betting/Football/',
        'betfair sportsbook': 'https://www.betfair.com/sport/football',
        'william hill': 'https://sports.williamhill.com/betting/en-gb'
    }

    if len(surebets) > 0:
        surebets['link'] = surebets['link'].map(bookie_url_mapping).fillna(surebets['link'])

    yesterday = datetime.datetime.now() - timedelta(days=1)
    yesterday_timestamp = datetime.datetime(yesterday.year, yesterday.month, yesterday.day, yesterday.hour, yesterday.minute, yesterday.second).timestamp()

    surebets['last_update'] = datetime.datetime.now().timestamp()
    df = pd.read_csv('csv_data/surebets.csv')
    df = df._append(surebets, ignore_index=True)
    df = df.loc[df['last_update'] > yesterday_timestamp].reset_index(drop=True)
    df['last_update'] = df['last_update'].astype(int)
    abc = []
    if len(df['last_update']) > 0:
        for x in range(0, len(df['last_update'])):
            last_update = df['last_update'][x]
            id = df['id'][x]
            abc.append(str(id).replace(r'[0123456789]', '') + str(id)[-2:] + str(last_update)[:10])
    df['id'] = abc
    df.sort_values(by='last_update', ascending=True)
    df.to_csv('csv_data/surebets.csv', index=False)

    if len(surebets) > 0:
        surebets = surebets.fillna('')
        surebets = surebets.loc[surebets['profit_in_%'] > 1].reset_index(drop=True)
        del surebets['scraped_date'], surebets['sport'], surebets['time'], surebets['home_team'], surebets['away_team'],
        surebets = surebets.rename(columns={'profit_in_%': '%', 'home': '1', 'draw': 'X', 'away': '2', 'b_score_y': 'bs_y', 'b_score_n': 'bs_n', 'first_g_1': 'g1_1', 'first_g_X': 'g1_X', 'first_g_2': 'g1_2', 'first_h_1': 'h1_1', 'first_h_X': 'h1_X', 'first_h_2': 'h1_2'})

    return surebets


def get_soccer_surebets_germany(events):

    df_surebets = pd.DataFrame()
    germanbookies = ['interwetten', 'betano', 'neobet', 'winamax', 'merkur_sports', 'bwin', 'ladbrokes', 'sportingbet',
                     'oddset', 'bpremium', 'betathome', 'pinnacle', 'dafabet', 'cloudbet', 'bet365', 'sport888', 'betway',
                     'bildbet', 'sportwetten_de', 'cbet', 'admiralbet', 'leonbet',  'mobilebet', 'tipico', 'chillybets',
                     'mybet', 'stake', '1bet', 'happybet', 'wettarena', 'wolfbet', 'tipwin', 'bet3000', 'bcgame', 'moonbet',
                     'roobet', 'duelbits', 'tiptorro', 'mystake', 'goldenbet', 'jackbit']
    events = events.loc[events['bookie'].isin(germanbookies)]
    for k in events['id'].unique():
        for i in range(0, 1):
            this_bet = events.loc[events['id'] == k]
            if len(this_bet) > 1:

                # HOME, DRAW, AWAY
                for ii in range(0, 1):
                    home = this_bet.loc[this_bet['home'] == max(this_bet['home'])].reset_index(drop=True)
                    draw = this_bet.loc[this_bet['draw'] == max(this_bet['draw'])].reset_index(drop=True)
                    away = this_bet.loc[this_bet['away'] == max(this_bet['away'])].reset_index(drop=True)

                    columns_to_reset = ['o45', 'u45', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15', 'u05', 'o05',
                                        'b_score_y', 'b_score_n', 'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1',
                                        'first_h_X', 'first_h_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1', 'hand02_X',
                                        'hand02_2', 'hand01_1', 'hand01_X', 'hand01_2', 'hand10_1', 'hand10_X', 'hand10_2',
                                        'hand20_1', 'hand20_X', 'hand20_2', 'hand30_1', 'hand30_X', 'hand30_2']
                    # Reset columns to None
                    home[columns_to_reset] = None
                    home['draw'] = None
                    home['away'] = None
                    draw[columns_to_reset] = None
                    draw['home'] = None
                    draw['away'] = None
                    away[columns_to_reset] = None
                    away['home'] = None
                    away['draw'] = None
                    if len(home) > 0 and len(draw) > 0 and len(away) > 0:
                        addition = home['p_home'][0] + draw['p_draw'][0] + away['p_away'][0]
                        home['eur'] = ((100 / addition) / home['home'][0]).round(1).astype(float)
                        draw['eur'] = ((100 / addition) / draw['draw'][0]).round(1).astype(float)
                        away['eur'] = ((100 / addition) / away['away'][0]).round(1).astype(float)

                        if home['p_home'][0] + draw['p_draw'][0] + away['p_away'][0] < 1.03:
                            df_surebet = pd.concat([home, draw, away], ignore_index=True)
                            df_surebet['profit_in_%'] = round(
                                100 / (home['p_home'][0] + draw['p_draw'][0] + away['p_away'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen

                # b_score_y, b_score_n
                for ii in range(0, 1):
                    b_score_y = this_bet.loc[this_bet['b_score_y'] == max(this_bet['b_score_y'])].reset_index(drop=True)
                    b_score_n = this_bet.loc[this_bet['b_score_n'] == max(this_bet['b_score_n'])].reset_index(drop=True)
                    columns_to_reset = ['home', 'draw', 'away', 'o45', 'u45', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15', 'u05', 'o05',
                                        'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1',
                                        'first_h_X', 'first_h_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1', 'hand02_X',
                                        'hand02_2', 'hand01_1', 'hand01_X', 'hand01_2', 'hand10_1', 'hand10_X', 'hand10_2',
                                        'hand20_1', 'hand20_X', 'hand20_2', 'hand30_1', 'hand30_X', 'hand30_2']
                    # Reset columns to None
                    b_score_y[columns_to_reset] = None
                    b_score_y['b_score_n'] = None
                    b_score_n[columns_to_reset] = None
                    b_score_n['b_score_y'] = None
                    if len(b_score_n) > 0 and len(b_score_y) > 0:
                        addition = b_score_y['p_b_score_y'][0] + b_score_n['p_b_score_n'][0]
                        b_score_y['eur'] = ((100 / addition) / b_score_y['b_score_y'][0]).round(1).astype(float)
                        b_score_n['eur'] = ((100 / addition) / b_score_n['b_score_n'][0]).round(1).astype(float)
                        if b_score_y['p_b_score_y'][0] + b_score_n['p_b_score_n'][0] < 1.03:
                            df_surebet = b_score_y._append(b_score_n, ignore_index=True)
                            df_surebet['profit_in_%'] = round(100 / (b_score_y['p_b_score_y'][0] + b_score_n['p_b_score_n'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen

                # o45, u45
                for ii in range(0, 1):
                    o45 = this_bet.loc[this_bet['o45'] == max(this_bet['o45'])].reset_index(drop=True)
                    u45 = this_bet.loc[this_bet['u45'] == max(this_bet['u45'])].reset_index(drop=True)
                    columns_to_reset = ['home', 'draw', 'away', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15',
                                        'u05', 'o05', 'b_score_y', 'b_score_n',
                                        'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1',
                                        'first_h_X', 'first_h_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1',
                                        'hand02_X',
                                        'hand02_2', 'hand01_1', 'hand01_X', 'hand01_2', 'hand10_1', 'hand10_X',
                                        'hand10_2',
                                        'hand20_1', 'hand20_X', 'hand20_2', 'hand30_1', 'hand30_X', 'hand30_2']
                    # Reset columns to None
                    o45[columns_to_reset] = None
                    o45['u45'] = None
                    u45[columns_to_reset] = None
                    u45['o45'] = None
                    if len(o45) > 0 and len(u45) > 0:
                        addition = o45['p_o45'][0] + u45['p_u45'][0]
                        o45['eur'] = ((100 / addition) / o45['o45'][0]).round(1).astype(float)
                        u45['eur'] = ((100 / addition) / u45['u45'][0]).round(1).astype(float)
                        if o45['p_o45'][0] + u45['p_u45'][0] < 1.03:
                            df_surebet = o45._append(u45, ignore_index=True)
                            df_surebet['profit_in_%'] = round(100 / (o45['p_o45'][0] + u45['p_u45'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen

                # o35, u35
                for ii in range(0, 1):
                    o45 = this_bet.loc[this_bet['o35'] == max(this_bet['o35'])].reset_index(drop=True)
                    u45 = this_bet.loc[this_bet['u35'] == max(this_bet['u35'])].reset_index(drop=True)
                    columns_to_reset = ['home', 'draw', 'away', 'o45', 'u45', 'u25', 'o25', 'o15', 'u15',
                                        'u05', 'o05', 'b_score_y', 'b_score_n',
                                        'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1',
                                        'first_h_X', 'first_h_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1',
                                        'hand02_X',
                                        'hand02_2', 'hand01_1', 'hand01_X', 'hand01_2', 'hand10_1', 'hand10_X',
                                        'hand10_2',
                                        'hand20_1', 'hand20_X', 'hand20_2', 'hand30_1', 'hand30_X', 'hand30_2']
                    # Reset columns to None
                    o45[columns_to_reset] = None
                    o45['u35'] = None
                    u45[columns_to_reset] = None
                    u45['o35'] = None
                    if len(o45) > 0 and len(u45) > 0:
                        addition = o45['p_o35'][0] + u45['p_u35'][0]
                        o45['eur'] = ((100 / addition) / o45['o35'][0]).round(1).astype(float)
                        u45['eur'] = ((100 / addition) / u45['u35'][0]).round(1).astype(float)
                        if o45['p_o35'][0] + u45['p_u35'][0] < 1.03:
                            df_surebet = o45._append(u45, ignore_index=True)
                            df_surebet['profit_in_%'] = round(100 / (o45['p_o35'][0] + u45['p_u35'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen

                # o25, u25
                for ii in range(0, 1):
                    o45 = this_bet.loc[this_bet['o25'] == max(this_bet['o25'])].reset_index(drop=True)
                    u45 = this_bet.loc[this_bet['u25'] == max(this_bet['u25'])].reset_index(drop=True)
                    columns_to_reset = ['home', 'draw', 'away', 'o45', 'u45', 'u35', 'o35', 'o15', 'u15',
                                        'u05', 'o05', 'b_score_y', 'b_score_n',
                                        'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1',
                                        'first_h_X', 'first_h_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1',
                                        'hand02_X',
                                        'hand02_2', 'hand01_1', 'hand01_X', 'hand01_2', 'hand10_1', 'hand10_X',
                                        'hand10_2',
                                        'hand20_1', 'hand20_X', 'hand20_2', 'hand30_1', 'hand30_X', 'hand30_2']
                    # Reset columns to None
                    o45[columns_to_reset] = None
                    o45['u25'] = None
                    u45[columns_to_reset] = None
                    u45['o25'] = None
                    if len(o45) > 0 and len(u45) > 0:
                        addition = o45['p_o25'][0] + u45['p_u25'][0]
                        o45['eur'] = ((100 / addition) / o45['o25'][0]).round(1).astype(float)
                        u45['eur'] = ((100 / addition) / u45['u25'][0]).round(1).astype(float)
                        if o45['p_o25'][0] + u45['p_u25'][0] < 1.03:
                            df_surebet = o45._append(u45, ignore_index=True)
                            df_surebet['profit_in_%'] = round(100 / (o45['p_o25'][0] + u45['p_u25'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen

                # o15, u15
                for ii in range(0, 1):
                    o45 = this_bet.loc[this_bet['o15'] == max(this_bet['o15'])].reset_index(drop=True)
                    u45 = this_bet.loc[this_bet['u15'] == max(this_bet['u15'])].reset_index(drop=True)
                    columns_to_reset = ['home', 'draw', 'away', 'o35', 'u35', 'u25', 'o25', 'o45', 'u45',
                                        'u05', 'o05', 'b_score_y', 'b_score_n',
                                        'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1',
                                        'first_h_X', 'first_h_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1',
                                        'hand02_X',
                                        'hand02_2', 'hand01_1', 'hand01_X', 'hand01_2', 'hand10_1', 'hand10_X',
                                        'hand10_2',
                                        'hand20_1', 'hand20_X', 'hand20_2', 'hand30_1', 'hand30_X', 'hand30_2']
                    # Reset columns to None
                    o45[columns_to_reset] = None
                    o45['u15'] = None
                    u45[columns_to_reset] = None
                    u45['o15'] = None
                    if len(o45) > 0 and len(u45) > 0:
                        addition = o45['p_o15'][0] + u45['p_u15'][0]
                        o45['eur'] = ((100 / addition) / o45['o15'][0]).round(1).astype(float)
                        u45['eur'] = ((100 / addition) / u45['u15'][0]).round(1).astype(float)
                        if o45['p_o15'][0] + u45['p_u15'][0] < 1.03:
                            df_surebet = o45._append(u45, ignore_index=True)
                            df_surebet['profit_in_%'] = round(100 / (o45['p_o15'][0] + u45['p_u15'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen

                # o05, u05
                for ii in range(0, 1):
                    o45 = this_bet.loc[this_bet['o05'] == max(this_bet['o05'])].reset_index(drop=True)
                    u45 = this_bet.loc[this_bet['u05'] == max(this_bet['u05'])].reset_index(drop=True)
                    columns_to_reset = ['home', 'draw', 'away', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15',
                                        'u45', 'o45', 'b_score_y', 'b_score_n',
                                        'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1',
                                        'first_h_X', 'first_h_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1',
                                        'hand02_X',
                                        'hand02_2', 'hand01_1', 'hand01_X', 'hand01_2', 'hand10_1', 'hand10_X',
                                        'hand10_2',
                                        'hand20_1', 'hand20_X', 'hand20_2', 'hand30_1', 'hand30_X', 'hand30_2']
                    # Reset columns to None
                    o45[columns_to_reset] = None
                    o45['u05'] = None
                    u45[columns_to_reset] = None
                    u45['o05'] = None
                    if len(o45) > 0 and len(u45) > 0:
                        addition = o45['p_o05'][0] + u45['p_u05'][0]
                        o45['eur'] = ((100 / addition) / o45['o05'][0]).round(1).astype(float)
                        u45['eur'] = ((100 / addition) / u45['u05'][0]).round(1).astype(float)
                        if o45['p_o05'][0] + u45['p_u05'][0] < 1.03:
                            df_surebet = o45._append(u45, ignore_index=True)
                            df_surebet['profit_in_%'] = round(100 / (o45['p_o05'][0] + u45['p_u05'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen

                # First Goal: HOME, DRAW, AWAY
                for ii in range(0, 1):
                    home = this_bet.loc[this_bet['first_g_1'] == max(this_bet['first_g_1'])].reset_index(drop=True)
                    draw = this_bet.loc[this_bet['first_g_X'] == max(this_bet['first_g_X'])].reset_index(drop=True)
                    away = this_bet.loc[this_bet['first_g_2'] == max(this_bet['first_g_2'])].reset_index(drop=True)
                    columns_to_reset = ['home', 'draw', 'away', 'o45', 'u45', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15', 'u05', 'o05',
                                        'b_score_y', 'b_score_n', 'first_h_1',
                                        'first_h_X', 'first_h_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1',
                                        'hand02_X',
                                        'hand02_2', 'hand01_1', 'hand01_X', 'hand01_2', 'hand10_1', 'hand10_X',
                                        'hand10_2',
                                        'hand20_1', 'hand20_X', 'hand20_2', 'hand30_1', 'hand30_X', 'hand30_2']
                    # Reset columns to None
                    home[columns_to_reset] = None
                    home['first_g_X'] = None
                    home['first_g_2'] = None
                    draw[columns_to_reset] = None
                    draw['first_g_1'] = None
                    draw['first_g_2'] = None
                    away[columns_to_reset] = None
                    away['first_g_1'] = None
                    away['first_g_X'] = None
                    if len(home) > 0 and len(draw) > 0 and len(away) > 0:
                        addition = home['p_first_g_1'][0] + draw['p_first_g_X'][0] + away['p_first_g_2'][0]
                        home['eur'] = ((100 / addition) / home['first_g_1'][0]).round(1).astype(float)
                        draw['eur'] = ((100 / addition) / draw['first_g_X'][0]).round(1).astype(float)
                        away['eur'] = ((100 / addition) / away['first_g_2'][0]).round(1).astype(float)
                        if home['p_first_g_1'][0] + draw['p_first_g_X'][0] + away['p_first_g_2'][0] < 1.03:
                            df_surebet = home._append(draw, ignore_index=True)
                            df_surebet = df_surebet._append(away, ignore_index=True)
                            df_surebet['profit_in_%'] = round(
                                100 / (home['p_first_g_1'][0] + draw['p_first_g_X'][0] + away['p_first_g_2'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen

                # First half: HOME, DRAW, AWAY
                for ii in range(0, 1):
                    home = this_bet.loc[this_bet['first_h_1'] == max(this_bet['first_h_1'])].reset_index(drop=True)
                    draw = this_bet.loc[this_bet['first_h_X'] == max(this_bet['first_h_X'])].reset_index(drop=True)
                    away = this_bet.loc[this_bet['first_h_2'] == max(this_bet['first_h_2'])].reset_index(drop=True)
                    columns_to_reset = ['home', 'draw', 'away', 'o45', 'u45', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15', 'u05', 'o05',
                                        'b_score_y', 'b_score_n', 'first_g_1', 'first_g_X', 'first_g_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1',
                                        'hand02_X',
                                        'hand02_2', 'hand01_1', 'hand01_X', 'hand01_2', 'hand10_1', 'hand10_X',
                                        'hand10_2',
                                        'hand20_1', 'hand20_X', 'hand20_2', 'hand30_1', 'hand30_X', 'hand30_2']
                    # Reset columns to None
                    home[columns_to_reset] = None
                    home['first_h_X'] = None
                    home['first_h_2'] = None
                    draw[columns_to_reset] = None
                    draw['first_h_1'] = None
                    draw['first_h_2'] = None
                    away[columns_to_reset] = None
                    away['first_h_1'] = None
                    away['first_h_X'] = None
                    if len(home) > 0 and len(draw) > 0 and len(away) > 0:
                        addition = home['p_first_h_1'][0] + draw['p_first_h_X'][0] + away['p_first_h_2'][0]
                        home['eur'] = ((100 / addition) / home['first_h_1'][0]).round(1).astype(float)
                        draw['eur'] = ((100 / addition) / draw['first_h_X'][0]).round(1).astype(float)
                        away['eur'] = ((100 / addition) / away['first_h_2'][0]).round(1).astype(float)
                        if home['p_first_h_1'][0] + draw['p_first_h_X'][0] + away['p_first_h_2'][0] < 1.03:
                            df_surebet = home._append(draw, ignore_index=True)
                            df_surebet = df_surebet._append(away, ignore_index=True)
                            df_surebet['profit_in_%'] = round(
                                100 / (home['p_first_h_1'][0] + draw['p_first_h_X'][0] + away['p_first_h_2'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen


                # HANDICAP 3:0
                for ii in range(0, 1):
                    home = this_bet.loc[this_bet['hand30_1'] == max(this_bet['hand30_1'])].reset_index(drop=True)
                    draw = this_bet.loc[this_bet['hand30_X'] == max(this_bet['hand30_X'])].reset_index(drop=True)
                    away = this_bet.loc[this_bet['hand30_2'] == max(this_bet['hand30_2'])].reset_index(drop=True)

                    columns_to_reset = ['home', 'draw', 'away', 'o45', 'u45', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15', 'u05', 'o05',
                                        'b_score_y', 'b_score_n', 'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1',
                                        'first_h_X', 'first_h_2', 'hand02_1', 'hand03_1', 'hand03_X', 'hand03_2',
                                        'hand02_X',
                                        'hand02_2', 'hand01_1', 'hand01_X', 'hand01_2', 'hand10_1', 'hand10_X',
                                        'hand10_2',
                                        'hand20_1', 'hand20_X', 'hand20_2']
                    # Reset columns to None
                    home[columns_to_reset] = None
                    home['hand30_X'] = None
                    home['hand30_2'] = None
                    draw[columns_to_reset] = None
                    draw['hand30_1'] = None
                    draw['hand30_2'] = None
                    away[columns_to_reset] = None
                    away['hand30_1'] = None
                    away['hand30_X'] = None
                    if len(home) > 0 and len(draw) > 0 and len(away) > 0:
                        addition = home['p_hand30_1'][0] + draw['p_hand30_X'][0] + away['p_hand30_2'][0]
                        home['eur'] = ((100 / addition) / home['hand30_1'][0]).round(1).astype(float)
                        draw['eur'] = ((100 / addition) / draw['hand30_X'][0]).round(1).astype(float)
                        away['eur'] = ((100 / addition) / away['hand30_2'][0]).round(1).astype(float)

                        if home['p_hand30_1'][0] + draw['p_hand30_X'][0] + away['p_hand30_2'][0] < 1.03:
                            df_surebet = pd.concat([home, draw, away], ignore_index=True)
                            df_surebet['profit_in_%'] = round(
                                100 / (home['p_hand30_1'][0] + draw['p_hand30_X'][0] + away['p_hand30_2'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen

                # HANDICAP 2:0
                for ii in range(0, 1):
                    home = this_bet.loc[this_bet['hand20_1'] == max(this_bet['hand20_1'])].reset_index(drop=True)
                    draw = this_bet.loc[this_bet['hand20_X'] == max(this_bet['hand20_X'])].reset_index(drop=True)
                    away = this_bet.loc[this_bet['hand20_2'] == max(this_bet['hand20_2'])].reset_index(drop=True)

                    columns_to_reset = ['home', 'draw', 'away', 'o45', 'u45', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15', 'u05', 'o05',
                                        'b_score_y', 'b_score_n', 'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1',
                                        'first_h_X', 'first_h_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1', 'hand02_X', 'hand02_2',
                                        'hand01_1', 'hand01_X', 'hand01_2', 'hand10_1', 'hand10_X',
                                        'hand10_2', 'hand30_1', 'hand30_X', 'hand30_2']
                    # Reset columns to None
                    home[columns_to_reset] = None
                    home['hand20_X'] = None
                    home['hand20_2'] = None
                    draw[columns_to_reset] = None
                    draw['hand20_1'] = None
                    draw['hand20_2'] = None
                    away[columns_to_reset] = None
                    away['hand20_1'] = None
                    away['hand20_X'] = None
                    if len(home) > 0 and len(draw) > 0 and len(away) > 0:
                        addition = home['p_hand20_1'][0] + draw['p_hand20_X'][0] + away['p_hand20_2'][0]
                        home['eur'] = ((100 / addition) / home['hand20_1'][0]).round(1).astype(float)
                        draw['eur'] = ((100 / addition) / draw['hand20_X'][0]).round(1).astype(float)
                        away['eur'] = ((100 / addition) / away['hand20_2'][0]).round(1).astype(float)

                        if home['p_hand20_1'][0] + draw['p_hand20_X'][0] + away['p_hand20_2'][0] < 1.03:
                            df_surebet = pd.concat([home, draw, away], ignore_index=True)
                            df_surebet['profit_in_%'] = round(
                                100 / (home['p_hand20_1'][0] + draw['p_hand20_X'][0] + away['p_hand20_2'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen

                # HANDICAP 1:0
                for ii in range(0, 1):
                    home = this_bet.loc[this_bet['hand10_1'] == max(this_bet['hand10_1'])].reset_index(drop=True)
                    draw = this_bet.loc[this_bet['hand10_X'] == max(this_bet['hand10_X'])].reset_index(drop=True)
                    away = this_bet.loc[this_bet['hand10_2'] == max(this_bet['hand10_2'])].reset_index(drop=True)

                    columns_to_reset = ['home', 'draw', 'away', 'o45', 'u45', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15', 'u05', 'o05',
                                        'b_score_y', 'b_score_n', 'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1',
                                        'first_h_X', 'first_h_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1', 'hand02_X', 'hand02_2',
                                        'hand01_1', 'hand01_X', 'hand01_2',
                                        'hand20_1', 'hand20_X', 'hand20_2', 'hand30_1', 'hand30_X', 'hand30_2']
                    # Reset columns to None
                    home[columns_to_reset] = None
                    home['hand10_X'] = None
                    home['hand10_2'] = None
                    draw[columns_to_reset] = None
                    draw['hand10_1'] = None
                    draw['hand10_2'] = None
                    away[columns_to_reset] = None
                    away['hand10_1'] = None
                    away['hand10_X'] = None
                    if len(home) > 0 and len(draw) > 0 and len(away) > 0:
                        addition = home['p_hand10_1'][0] + draw['p_hand10_X'][0] + away['p_hand10_2'][0]
                        home['eur'] = ((100 / addition) / home['hand10_1'][0]).round(1).astype(float)
                        draw['eur'] = ((100 / addition) / draw['hand10_X'][0]).round(1).astype(float)
                        away['eur'] = ((100 / addition) / away['hand10_2'][0]).round(1).astype(float)

                        if home['p_hand10_1'][0] + draw['p_hand10_X'][0] + away['p_hand10_2'][0] < 1.03:
                            df_surebet = pd.concat([home, draw, away], ignore_index=True)
                            df_surebet['profit_in_%'] = round(
                                100 / (home['p_hand10_1'][0] + draw['p_hand10_X'][0] + away['p_hand10_2'][0]) - 100, 2)
                            if max(df_surebet['profit_in_%']) > 30:
                                break
                            df_surebets = df_surebets._append(df_surebet, ignore_index=True)
                            # hier noch weitere infrage kommende bookies hinzufügen


    if len(df_surebets) > 0:
        df_surebets = df_surebets.sort_values(by=['profit_in_%', 'id'], ascending=False)
        df_surebets = df_surebets.drop_duplicates(subset=['bookie', 'id', 'eur'], keep='last')
        df_surebets = df_surebets.loc[(df_surebets['profit_in_%'] > 0.1) & (df_surebets['profit_in_%'] < 22)].reset_index(drop=True)
        df_surebets['link'] = df_surebets['url']
        del df_surebets['country'], df_surebets['p_home'], df_surebets['p_draw'], df_surebets['p_away'], df_surebets['p_b_score_y'], df_surebets['p_b_score_n'], df_surebets['p_o45'], df_surebets['p_u45'], df_surebets['p_o35'], df_surebets['p_u35']
        del df_surebets['url'], df_surebets['p_o25'], df_surebets['p_u25'], df_surebets['p_o15'], df_surebets['p_u15'], df_surebets['p_o05'], df_surebets['p_u05']
        del df_surebets['p_first_g_1'], df_surebets['p_first_g_X'], df_surebets['p_first_g_2'], df_surebets['p_first_h_1'], df_surebets['p_first_h_X'], df_surebets['p_first_h_2']
        del df_surebets['p_hand30_1'], df_surebets['p_hand30_X'], df_surebets['p_hand30_2'], df_surebets['p_hand20_1'], df_surebets['p_hand20_X'], df_surebets['p_hand20_2'], df_surebets['p_hand10_1'], df_surebets['p_hand10_X'], df_surebets['p_hand10_2']
    return df_surebets


def surebets_preprocessing_germany(surebets_ger, c):

    surebets = surebets_ger.copy()
    bookie_url_mapping = {
        'marathon bet': 'https://www.marathonbet.com/en/betting/Football/',
        'betfair sportsbook': 'https://www.betfair.com/sport/football',
        'william hill': 'https://sports.williamhill.com/betting/en-gb'
    }

    if len(surebets) > 0:
        surebets['link'] = surebets['link'].map(bookie_url_mapping).fillna(surebets['link'])

    yesterday = datetime.datetime.now() - timedelta(days=1)
    yesterday_timestamp = datetime.datetime(yesterday.year, yesterday.month, yesterday.day, yesterday.hour, yesterday.minute, yesterday.second).timestamp()


    surebets['last_update'] = datetime.datetime.now().timestamp()
    '''
    df = pd.read_csv('csv_data/surebets.csv')
    df = df._append(surebets, ignore_index=True)
    df = df.loc[df['last_update'] > yesterday_timestamp].reset_index(drop=True)
    df['last_update'] = df['last_update'].astype(int)
    abc = []
    if len(df['last_update']) > 0:
        for x in range(0, len(df['last_update'])):
            last_update = df['last_update'][x]
            id = df['id'][x]
            abc.append(str(id).replace(r'[0123456789]', '') + str(id)[-2:] + str(last_update)[:10])
    df['id'] = abc
    df.sort_values(by='last_update', ascending=True)
    df.to_csv('csv_data/surebets.csv', index=False)
    '''

    if len(surebets) > 0:
        surebets = surebets.fillna('')
        surebets = surebets.loc[surebets['profit_in_%'] > 1].reset_index(drop=True)
        del surebets['scraped_date'], surebets['sport'], surebets['time'], surebets['home_team'], surebets['away_team'],
        surebets = surebets.rename(columns={'profit_in_%': '%', 'home': '1', 'draw': 'X', 'away': '2', 'b_score_y': 'bs_y', 'b_score_n': 'bs_n', 'first_g_1': 'g1_1', 'first_g_X': 'g1_X', 'first_g_2': 'g1_2', 'first_h_1': 'h1_1', 'first_h_X': 'h1_X', 'first_h_2': 'h1_2'})

    return surebets


def get_soccer_valuebets(events):

    events = events.loc[(events['bookie'] != 'fertik') & (events['bookie'] != 'fertig')]
    events = events.drop_duplicates()

    for k in range(0, 1):
        events['p_matchwinner'] = events['p_home'] + events['p_draw'] + events['p_away']
        events['true_p_home'] = events['p_home'] / events['p_matchwinner']
        events['true_p_draw'] = events['p_draw'] / events['p_matchwinner']
        events['true_p_away'] = events['p_away'] / events['p_matchwinner']

        events['p_b_score'] = events['p_b_score_y'] + events['p_b_score_n']
        events['true_p_b_score_y'] = events['p_b_score_y'] / events['p_b_score']
        events['true_p_b_score_n'] = events['p_b_score_n'] / events['p_b_score']

        events['p_ou45'] = events['p_o45'] + events['p_u45']
        events['true_p_o45'] = events['p_o45'] / events['p_ou45']
        events['true_p_u45'] = events['p_u45'] / events['p_ou45']

        events['p_ou35'] = events['p_o35'] + events['p_u35']
        events['true_p_o35'] = events['p_o35'] / events['p_ou35']
        events['true_p_u35'] = events['p_u35'] / events['p_ou35']

        events['p_ou25'] = events['p_o25'] + events['p_u25']
        events['true_p_o25'] = events['p_o25'] / events['p_ou25']
        events['true_p_u25'] = events['p_u25'] / events['p_ou25']

        events['p_ou15'] = events['p_o15'] + events['p_u15']
        events['true_p_o15'] = events['p_o15'] / events['p_ou15']
        events['true_p_u15'] = events['p_u15'] / events['p_ou15']

        events['p_ou05'] = events['p_o05'] + events['p_u05']
        events['true_p_o05'] = events['p_o05'] / events['p_ou05']
        events['true_p_u05'] = events['p_u05'] / events['p_ou05']

        events['p_first_g'] = events['p_first_g_1'] + events['p_first_g_X'] + events['p_first_g_2']
        events['true_p_first_g_1'] = events['p_first_g_1'] / events['p_first_g']
        events['true_p_first_g_X'] = events['p_first_g_X'] / events['p_first_g']
        events['true_p_first_g_2'] = events['p_first_g_2'] / events['p_first_g']

        events['p_first_h'] = events['p_first_h_1'] + events['p_first_h_X'] + events['p_first_h_2']
        events['true_p_first_h_1'] = events['p_first_h_1'] / events['p_first_h']
        events['true_p_first_h_X'] = events['p_first_h_X'] / events['p_first_h']
        events['true_p_first_h_2'] = events['p_first_h_2'] / events['p_first_h']

        events['p_hand30'] = events['p_hand30_1'] + events['p_hand30_X'] + events['p_hand30_2']
        events['true_p_hand30_1'] = events['p_hand30_1'] / events['p_hand30']
        events['true_p_hand30_X'] = events['p_hand30_X'] / events['p_hand30']
        events['true_p_hand30_2'] = events['p_hand30_2'] / events['p_hand30']

        events['p_hand20'] = events['p_hand20_1'] + events['p_hand20_X'] + events['p_hand20_2']
        events['true_p_hand20_1'] = events['p_hand20_1'] / events['p_hand20']
        events['true_p_hand20_X'] = events['p_hand20_X'] / events['p_hand20']
        events['true_p_hand20_2'] = events['p_hand20_2'] / events['p_hand20']

        events['p_hand10'] = events['p_hand10_1'] + events['p_hand10_X'] + events['p_hand10_2']
        events['true_p_hand10_1'] = events['p_hand10_1'] / events['p_hand10']
        events['true_p_hand10_X'] = events['p_hand10_X'] / events['p_hand10']
        events['true_p_hand10_2'] = events['p_hand10_2'] / events['p_hand10']

    df_valuebets = pd.DataFrame()
    for k in events['match'].unique():
        print(k)
        for i in range(0, 1):
            this_bet = events.loc[events['match'] == k].reset_index(drop=True)
            this_bet['date'].fillna(method='ffill', inplace=True)
            this_bet['date'].fillna(method='bfill', inplace=True)
            this_bet['time'].fillna(method='ffill', inplace=True)
            this_bet['time'].fillna(method='bfill', inplace=True)
            if len(this_bet) > 24:
                true_probs = ['true_p_home', 'true_p_draw', 'true_p_away', 'true_p_b_score_y', 'true_p_b_score_n',
                              # 'true_p_o45', 'true_p_u45',
                              'true_p_o35', 'true_p_u35', 'true_p_o25', 'true_p_u25', 'true_p_o15', 'true_p_u15',
                              # 'true_p_o05', 'true_p_u05',
                              'true_p_first_g_1', 'true_p_first_g_X', 'true_p_first_g_2', 'true_p_first_h_1',
                              'true_p_first_h_X', 'true_p_first_h_2',
                              'true_p_hand30_1', 'true_p_hand30_X', 'true_p_hand30_2', 'true_p_hand20_1',
                              'true_p_hand20_X', 'true_p_hand20_2',
                              'true_p_hand10_1', 'true_p_hand10_X', 'true_p_hand10_2']
                probs = ['p_home', 'p_draw', 'p_away', 'p_b_score_y', 'p_b_score_n',  # 'p_o45', 'p_u45',
                         'p_o35', 'p_u35', 'p_o25', 'p_u25', 'p_o15', 'p_u15',  # 'p_o05', 'p_u05',
                         'p_first_g_1', 'p_first_g_X', 'p_first_g_2', 'p_first_h_1', 'p_first_h_X', 'p_first_h_2',
                         'p_hand30_1', 'p_hand30_X', 'p_hand30_2', 'p_hand20_1', 'p_hand20_X', 'p_hand20_2',
                         'p_hand10_1', 'p_hand10_X', 'p_hand10_2']
                for c in range(0, len(true_probs)):
                    market_prop_titel = true_probs[c]
                    bookie_prob_titel = probs[c]
                    pre_l = this_bet[str(market_prop_titel)]
                    pre_l = pre_l.dropna().reset_index(drop=True)
                    l_all = np.array(list(set(pre_l)))
                    if len(l_all) > 2:
                        statistic, p_value = stats.shapiro(l_all)
                        alpha = 0.1
                        if p_value > alpha:  # Check if the p-value is less than the significance level
                            # Calculate the IQR (Interquartile Range)
                            q1 = np.percentile(l_all, 25)
                            q3 = np.percentile(l_all, 75)
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            l = l_all[(l_all >= lower_bound) & (l_all <= upper_bound)]
                            if len(l) > 16:
                                fair_market_odd = (sum(l) / len(l)) * 0.98
                                shorter_bet = this_bet
                                shorter_bet['fair_market_prop'] = fair_market_odd
                                shorter_bet = shorter_bet.dropna(subset=bookie_prob_titel).reset_index(drop=True)
                                shorter_bet['value'] = shorter_bet['fair_market_prop'] * shorter_bet[
                                    str(market_prop_titel).replace('true_p_', '')].astype(float)
                                shorter_bet['ev_profit_in_%'] = (shorter_bet['value'] - 1) * 100
                                shorter_bet = shorter_bet.loc[shorter_bet['value'] >= 1].reset_index(drop=True)
                                for s in range(0, len(shorter_bet[str(bookie_prob_titel)])):
                                    df_valuebet = pd.DataFrame()
                                    ev_bet = shorter_bet.loc[shorter_bet.index == s].reset_index(drop=True)
                                    plt.hist(l_all, bins=12, edgecolor='black')
                                    print(ev_bet)
                                    print('fair_market_odd', fair_market_odd, bookie_prob_titel,
                                          this_bet[str(bookie_prob_titel)].mean())

                                    df_valuebet['bookie'] = ev_bet['bookie']
                                    df_valuebet['sport'] = ev_bet['sport']
                                    df_valuebet['country'] = ev_bet['country']
                                    df_valuebet['home_team'] = ev_bet['home_team']
                                    df_valuebet['away_team'] = ev_bet['away_team']
                                    df_valuebet['bettype'] = str(bookie_prob_titel).replace('p_', '').replace('_bookie',
                                                                                                              '')
                                    df_valuebet['marketodd'] = round(1 / fair_market_odd, 3) * 1.0
                                    df_valuebet['bookieodd'] = 1 / ev_bet[str(bookie_prob_titel).replace('true_p_', '')]
                                    df_valuebet['p_market'] = round(fair_market_odd, 3)
                                    df_valuebet['p_bookie'] = np.round(ev_bet[str(bookie_prob_titel)], 3)
                                    df_valuebet['risk_in_%'] = round(np.std(l) * 10000, 2)
                                    print(bookie_prob_titel, fair_market_odd, market_prop_titel)
                                    df_valuebet['ev_profit_in_%'] = np.round(ev_bet['ev_profit_in_%'], 2)
                                    # df_valuebet['ev_profit_in_%'] = df_valuebet['ev_profit_in_%'] - 1
                                    #df_valuebet['ev_profit_in_%'] = np.round(df_valuebet['ev_profit_in_%'] * df_valuebet['p_market'], 4) * 100df_valuebet['date'] = ev_bet['date']
                                    df_valuebet['match'] = ev_bet['match']
                                    #df_valuebet['id'] = ev_bet['id']
                                    df_valuebet['scraped_date'] = ev_bet['scraped_date']
                                    df_valuebet['date'] = ev_bet['date']
                                    df_valuebet['time'] = ev_bet['time']
                                    df_valuebet['link'] = ev_bet['url']
                                    df_valuebets = df_valuebets._append(df_valuebet, ignore_index=True)


    if len(df_valuebets) > 0:
        df_valuebets = df_valuebets.sort_values(by=['ev_profit_in_%', 'match'], ascending=False)
        df_valuebets = df_valuebets.drop_duplicates(subset=['bookie', 'match', 'bettype'], keep='last')
        df_valuebets = df_valuebets.loc[(df_valuebets['ev_profit_in_%'] > 0.3) & (df_valuebets['ev_profit_in_%'] < 8) & (df_valuebets['risk_in_%'] < 200)].reset_index(drop=True)
    print(df_valuebets)

    return df_valuebets


def valuebets_preprocessing(valuebets, c):
    bookie_url = []
    if len(valuebets) > 0:
        for i in valuebets['link']:
            if i == 'marathon bet':
                bookie_url.append('https://www.marathonbet.com/en/betting/Football/')
            elif i == 'betfair sportsbook':
                bookie_url.append('https://www.betfair.com/sport/football')
            elif i == 'william hill':
                bookie_url.append('https://sports.williamhill.com/betting/en-gb')
            else:
                bookie_url.append(i)
        valuebets['link'] = bookie_url

    if len(valuebets) > 0:
        profit = []
        for v in valuebets['ev_profit_in_%']:
            if len(str(v)) > 5:
                profit.append(float(str(v)[:5]))
            else:
                profit.append(float(str(v)))
        valuebets['ev_profit_in_%'] = profit

    yesterday = datetime.datetime.now() - timedelta(days=20)
    yesterday_timestamp = datetime.datetime(yesterday.year, yesterday.month, yesterday.day, yesterday.hour, yesterday.minute, yesterday.second).timestamp()

    valuebets['last_update'] = datetime.datetime.now().timestamp()
    df = pd.read_csv('csv_data/valuebets.csv')
    df = df._append(valuebets, ignore_index=True)
    df = df.loc[df['last_update'] > yesterday_timestamp].reset_index(drop=True)
    df['last_update'] = df['last_update'].astype(int)
    abc = []
    if len(df['last_update']) > 0:
        for x in range(0, len(df['last_update'])):
            last_update = df['last_update'][x]
            id = df['match'][x]
            abc.append(str(id).replace(r'[0123456789]', '') + str(id)[-2:] + str(last_update)[:10])
    #df['match'] = abc
    df.sort_values(by='last_update', ascending=True)
    df.to_csv('csv_data/valuebets.csv', index=False)

    if len(valuebets) > 0:
        valuebets = valuebets.fillna('')
        twitter.post_valuebets_twitter(valuebets, c)
        del valuebets['scraped_date'], valuebets['sport'], valuebets['time'], valuebets['home_team'], valuebets['away_team'], valuebets['country']
        valuebets = valuebets.loc[valuebets['ev_profit_in_%'] > 1.5].reset_index(drop=True)
        valuebets = valuebets.rename(columns={'ev_profit_in_%': 'exp_%', 'home': '1', 'draw': 'X', 'away': '2', 'b_score_y': 'bs_y', 'b_score_n': 'bs_n', 'first_g_1': 'g1_1', 'first_g_X': 'g1_X', 'first_g_2': 'g1_2', 'first_h_1': 'h1_1', 'first_h_X': 'h1_X', 'first_h_2': 'h1_2'})

    return valuebets


def portfolio():

    valuebets = pd.read_csv('csv_data/valuebets.csv')
    portfolio = pd.read_csv('csv_data/portfolio.csv').sort_values(by='match')
    valuebets = valuebets.loc[valuebets['bettype'].isin(['home', 'draw', 'away', 'b_score_y', 'b_score_n', 'o15', 'u15', 'o25', 'u25', 'o35', 'u35'])].reset_index(drop=True)
    valuebets = valuebets.loc[(valuebets['ev_profit_in_%'] > 1) & (valuebets['risk_in_%'] < 150) & (valuebets['bookieodd'] <= 10)].reset_index(drop=True)
    valuebets = valuebets.drop_duplicates(subset=['bookie', 'bettype', 'match']).reset_index(drop=True).sort_values(by='match')

    valuebets = valuebets.loc[~((valuebets['match'].isin(portfolio['match'])) & (valuebets['bookie'].isin(portfolio['bookie'])))] #?
    print(len(valuebets))
    print(valuebets.tail())

    for h in range(0, len(valuebets)):
        #break
        assets_today = portfolio.sort_values(by='date').reset_index(drop=True)['assets'].values[-1] #last_update
        last_update = valuebets['last_update'].values[h]
        bettype = valuebets['bettype'].values[h]
        bookieodd = valuebets['bookieodd'].values[h]
        fairodd = valuebets['marketodd'].values[h]
        match = valuebets['match'].values[h]
        bookie = valuebets['bookie'].values[h]
        date = valuebets['date'].values[h]
        prevmatch = valuebets['match'].values[h - 1]
        if h > 0:
            if len(str(date).split('-')) == 3:
                if len(str(date).split('-')[0]) == 4:
                    given_date = datetime.datetime.strptime(str(date), "%Y-%m-%d")
                    current_datetime = datetime.datetime.now()
                    if given_date <= current_datetime:
                        print('GAME OVER:', match, date)
                        if 'vs' in match:
                            df_this_game = pd.DataFrame()
                            time.sleep(0.2)
                            home_team = match.split('vs')[0]
                            away_team = match.split('vs')[1]

                            home_score = -2
                            away_score = -1
                            team1 = 'hubbabubba'
                            team2 = 'bananenmann'
                            if match != prevmatch:
                                'New sofascore request was done!'
                                #https://lmt.fn.sportradar.com/common/de/Etc:UTC/gismo/match_timelinedelta/42726005
                                url = "https://sofascores.p.rapidapi.com/v1/search/multi"
                                querystring = {"query": str(home_team), "group": "teams"}
                                response = requests.get(url, headers=headers, params=querystring)
                                jdata = response.json()
                                try:
                                    if jdata['data'][0]['sport']['id'] == 1:
                                        team_id = jdata['data'][0]['id']
                                        url = "https://sofascores.p.rapidapi.com/v1/teams/events"
                                        querystring = {"page": "0", "course_events": "last", "team_id": str(team_id)}
                                        response = requests.get(url, headers=headers, params=querystring)
                                        jdata2 = response.json()

                                        team1 = jdata2['data']['events'][-1]['homeTeam']['name'].lower()
                                        team2 = jdata2['data']['events'][-1]['awayTeam']['name'].lower()

                                        seq = difflib.SequenceMatcher(None, team1, home_team)
                                        d_home_teams = seq.ratio() * 100
                                        seq = difflib.SequenceMatcher(None, team2, away_team)
                                        d_away_teams = seq.ratio() * 100
                                        if d_home_teams > 65:
                                            if d_away_teams > 65:
                                                home_score = jdata2['data']['events'][-1]['homeScore']['current']
                                                away_score = jdata2['data']['events'][-1]['awayScore']['current']
                                except: a = 1

                            else:
                                last_done_game = portfolio.tail(1).reset_index(drop=True)
                                if last_done_game['match'][0] == match:
                                    team1 = last_done_game['home_team'][0]
                                    team2 = last_done_game['away_team'][0]
                                    home_score = last_done_game['home_score'][0]
                                    away_score = last_done_game['away_score'][0]

                            if team1 != 'hubbabubba' and home_score >= 0:
                                homedrawaway = ['home', 'draw', 'away']
                                bscoreybscoren = ['b_score_y', 'b_score_n']
                                ou45 = ['o45', 'u45']
                                ou35 = ['o35', 'u35']
                                ou25 = ['o25', 'u25']
                                ou15 = ['o15', 'u15']
                                ou05 = ['o05', 'u05']
                                results_game = ''
                                if bettype in homedrawaway:
                                    if int(home_score) == int(away_score):
                                        results_game = 'draw'
                                    elif int(home_score) > int(away_score):
                                        results_game = 'home'
                                    else:
                                        results_game = 'away'
                                elif bettype in bscoreybscoren:
                                    if int(home_score) > 0 and int(away_score) > 0:
                                        results_game = 'b_score_y'
                                    else:
                                        results_game = 'b_score_n'
                                elif bettype in ou45:
                                    if int(home_score) + int(away_score) > 4:
                                        results_game = 'o45'
                                    else:
                                        results_game = 'u45'
                                elif bettype in ou35:
                                    if int(home_score) + int(away_score) > 3:
                                        results_game = 'o35'
                                    else:
                                        results_game = 'u35'
                                elif bettype in ou25:
                                    if int(home_score) + int(away_score) > 2:
                                        results_game = 'o25'
                                    else:
                                        results_game = 'u25'
                                elif bettype in ou15:
                                    if int(home_score) + int(away_score) > 1:
                                        results_game = 'o15'
                                    else:
                                        results_game = 'u15'
                                elif bettype in ou05:
                                    if int(home_score) + int(away_score) > 0:
                                        results_game = 'o05'
                                    else:
                                        results_game = 'u05'


                                if bettype == results_game:
                                    bet_right = 1
                                else:
                                    bet_right = 0
                                if bet_right == 1:
                                    profit = float(bookieodd) - 1
                                else:
                                    profit = -1

                                df_this_game['home_team'] = [team1]
                                df_this_game['away_team'] = [team2]
                                df_this_game['home_score'] = [home_score]
                                df_this_game['away_score'] = [away_score]
                                df_this_game['game_over'] = [1]
                                df_this_game['bookie'] = [bookie]
                                df_this_game['match'] = [match]
                                df_this_game['last_update'] = [last_update]
                                df_this_game['fairodd'] = [fairodd]
                                df_this_game['bookieodd'] = [bookieodd]
                                df_this_game['bettype'] = [bettype]
                                df_this_game['results_game'] = [results_game]
                                df_this_game['bet_right'] = [bet_right]
                                df_this_game['assets_before'] = [float(assets_today)]
                                df_this_game['assets'] = [float(assets_today) + float(assets_today) * 0.005 * profit]
                                df_this_game['date'] = datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
                                print(df_this_game)

                                portfolio = portfolio._append(df_this_game, ignore_index=True)
                                portfolio = portfolio.drop_duplicates(subset=['match', 'bookie', 'date'])
                                portfolio.to_csv('csv_data/portfolio.csv', index=False)
                                time.sleep(1)
                    else:
                        print('Match has not started yet!', match, given_date, current_datetime)


    df = pd.read_csv('csv_data/portfolio.csv').sort_values(by='date')
    df = df.drop_duplicates(subset='date', keep='last')
    fig = px.line(df, x='date', y="assets",
                  title='Virtual Portfolio for our Valuebets',
                  labels={
                      "assets": "account volume (€)"
                  })
    fig.update_traces(line=dict(color="#48adff", width=2))
    fig.update_layout(

        plot_bgcolor='#F8F8F8',
        font_family="Arial",
        title_font_family="Arial",
        font=dict(
            family="Arial",
            size=12,
            color='#071d44'
        ),
        title=dict(
            text="Virtual Portfolio for our Valuebets",
            x=0.5,
            y=0.94,
            font=dict(
                family="Arial",
                size=16,
                color='#071d44')
        )
    )
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=False,
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=False,
        gridcolor='lightgrey'
    )
    fig.write_html('csv_data/portfolio.html')
    plt.savefig('csv_data/portfolio.png')


def send_mails(surebets, valuebets, recievers):

    msg = MIMEMultipart()
    msg['From'] = 'stockinsightco@gmail.com'
    msg['Subject'] = 'Surebets / Valuebets Alert - Soccer'
    del surebets['last_update']

    if len(surebets) > 0:
        for x in surebets['id'].unique():
            surebetsx = surebets.loc[surebets['id'] == x]
            non_deletable_cols = []
            for col in surebetsx.columns:
                uniq = surebetsx[col].unique().tolist()
                uu = []
                for u in uniq:
                    if u != '':
                        uu.append(u)
                if len(uu) > 0:
                    non_deletable_cols.append(col)
            surebetsx = surebetsx[non_deletable_cols]
            #del surebetsx['id']
            if len(surebetsx) > 0:
                html = """\
                <html>
                  <body>
                    {0}
                    <br>
                    </br>
                  </body>
                </html>
                """.format(surebetsx.to_html())
                msg.attach(MIMEText(html, 'html'))

        if len(valuebets) > 0:
            html = """\
            <html>
                <body>
                Valuebets: \n
                {0}
                <br>
                </br>
              </body>
            </html>
            """.format(valuebets.to_html())
            msg.attach(MIMEText(html, 'html'))


        mailserver = smtplib.SMTP('smtp.gmail.com', 587)
        # identify ourselves to smtp gmail client
        mailserver.ehlo()
        # secure our email with tls encryption
        mailserver.starttls()
        # re-identify ourselves as an encrypted cnnection
        mailserver.ehlo()
        mailserver.login(msg['From'], 'mwssmhornzfblkxo')
        for to in recievers:
            mailserver.sendmail(msg['From'], to, msg.as_string())
        mailserver.quit()
        print('Mail got sent')


main()