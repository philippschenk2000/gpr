import datetime
import difflib
import random
import re
import smtplib
import time
import zoneinfo
from flask import jsonify
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
import bookies.gamebookers as gamebookers
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
import bookies.yonibet as yonibet
import bookies.casinozer as casinozer
import bookies.daznbet as daznbet
#import twitter
import tester
from selenium.webdriver.chrome.options import Options
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# just soccer, ('/api/v2/bookmakers')
def soccer_bookmakers():
    bookmakers = ['interwetten', 'dreambet', 'betobet', 'betano', 'winamax', 'merkur_sports', 'bwin',
                  'ladbrokes', 'sportingbet', 'oddset', 'bpremium', 'betathome', 'pinnacle', 'dafabet', 'cloudbet',
                  'bet365', 'daznbet',
                  'sport888', 'betway', 'n1bet', 'zebet', 'megapari', 'paripesa', '1xbit', 'bildbet', 'sportwetten_de',
                  'evobet', 'expekt', 'leovegas', 'casumo', 'betvictor', 'nextbet',
                  '22bet', 'netbet', 'vave', '20bet', 'ivibet', 'pokerstars', 'unibet', 'cbet', 'pmu', 'admiralbet',
                  'leonbet', 'lsbet', 'mobilebet', 'cricbaba', 'jet10', 'sultanbet', 'lilibet', 'fdj', 'betmaster',
                  'tipico', 'chillybets', '1xbet',
                  'gastonred', 'librabet', 'campeonbet', 'alphawin', 'kto', 'fezbet', 'powbet', 'sportaza', 'sgcasino',
                  'betsamigo', 'greatwin', 'betplay', 'joabet', 'mybet', 'stake', 'gamebookers', '1bet', 'quickwin',
                  'bankonbet', 'happybet', 'wettarena', 'bambet', 'qbet', 'cobrabet', 'rocketplay', 'winz', 'betibet',
                  'virginbet', 'livescorebet', 'skybet', 'playzilla', 'weltbet', 'betrophy', 'olympusbet', 'nucleonbet',
                  'dachbet', 'cashalot', 'zotabet', 'winning', 'rabonabet', 'wazamba', 'betstro', '32red', 'vistabet',
                  'wolfbet', 'betclic', 'tipwin', 'bet3000', 'bcgame', 'betfury', 'bluechip', 'gamblingapes',
                  'joycasino', 'moonbet', 'yonibet', 'casinozer',
                  'nearcasino', 'owlgames', 'rajbets', 'roobet', 'solcasino', 'rollbit', 'terracasino', 'duelbits',
                  'lottoland', 'magicalvegas', 'tiptorro', 'genybet', 'mystake', 'freshbet', 'draftkings', 'fanduel',
                  'betuk', 'parimatch',
                  'goldenbet', 'jackbit', '31bet', 'betmgm', 'caesars', 'ps3838', 'piwi247', 'asianodds']
    return bookmakers


# for all sports, ('/api/v2/allsports')
def getallsports():
    df_sports = pd.read_csv('csv_data/sports.csv').sort_values(by=['sports'], ascending=True).reset_index(drop=True).transpose().fillna(value='')
    return df_sports.to_dict()


# for all sports, ('/api/v2/competitions')
def getcompetitions(sport):
    df_competitions = pd.read_csv(f'csv_data/{sport}/leagues.csv')
    df_competitions = df_competitions[['sports', 'country', 'competition']].drop_duplicates()
    return df_competitions.sort_values(by=['sports', 'country', 'competition'], ascending=True).reset_index(drop=True).transpose().fillna(value='').to_dict()


# for all sports, ('/api/v2/matches')
def getmatches(sport, country, competition, match_urls_key):    # checking 1 time per day, abspeichern in csv wenn heute schon + api urls / urls

    # Once per day checking
    dates = pd.read_csv('csv_data/date.csv')
    df1 = dates.loc[(dates['sport'] == sport) & (dates['country'] == country) & (dates['competition'] == competition)]
    if str(datetime.datetime.now(tz=zoneinfo.ZoneInfo(key='Europe/Berlin')).strftime('%Y-%m-%d')) in df1['date'].tolist():
        print('URLs already saved')
    else:
        date = pd.DataFrame()
        date['date'] = [str(datetime.datetime.now(tz=zoneinfo.ZoneInfo(key='Europe/Berlin')).strftime('%Y-%m-%d'))]
        date['sport'] = [sport]
        date['country'] = [country]
        date['competition'] = [competition]
        dates = dates._append(date, ignore_index=True)
        dates.to_csv('csv_data/date.csv', index=False)#     !!!!!!!!!!!!!!!! entferne #

        # Collect the new matches, dates, times & api urls / urls
        df_competition = pd.read_csv(f'csv_data/{sport}/leagues.csv')      # !!!!!!!!!!!!!!!! entferne head
        df_competition = df_competition.loc[(df_competition['sports'] == sport) & (df_competition['country'] == country) & (
                        df_competition['competition'] == competition)].reset_index(drop=True)
        chrome_bookies = ['interwetten', 'admiralbet', 'merkur_sports', 'winamax', 'joabet', 'tipwin', 'daznbet',
                          'genybet', 'pmu', 'zebet', 'dreambet', 'mobilebet', 'netbet', 'leonbet', 'wolfbet', 'cbet', 'betathome']
        print(df_competition)
        api_competitions = df_competition.loc[~df_competition['bookie'].isin(chrome_bookies)].reset_index(drop=True)
        chrome_competitions = df_competition.loc[df_competition['bookie'].isin(chrome_bookies)].reset_index(drop=True)
        matches_api = getmatches_api(api_competitions)
        matches_chrome = getmatches_chrome(chrome_competitions)
        matches = matches_api._append(matches_chrome, ignore_index=True)
        matches = same_events_preprocessing(matches)
        same_events(matches, sport, country, competition)

    # things for return, füge die bookies pro game zusammen, genau wie die match-urls
    same_matches = pd.read_csv(f'csv_data/{sport}/matched4{country}_{competition}.csv')
    same_matches = preprocessing_for_return(same_matches, match_urls_key)
    return same_matches.transpose().fillna(value='').to_dict()    # returns everything except the api urls / urls


# for all sports
def getmatches_api(api_competitions):

    if len(api_competitions) > 0:
        sports = api_competitions['sports'].values[0]
        country = api_competitions['country'].values[0]
        competition = api_competitions['competition'][0]
        matches = pd.DataFrame()
        for k in range(0, len(api_competitions)):
            #break
            url = api_competitions['url'].values[k]
            print('in progress:', url)
            bookie = api_competitions['bookie'].values[k]
            try:
                match = ''
                if sports == 'soccer':
                    if bookie == 'betano':
                        match = betano.soccer_matches(sports, country, bookie, competition, url)
                    elif bookie == 'wettarena':
                        match = wettarena.soccer_matches(sports, country, bookie, competition, url)
                    elif bookie == 'bet3000':
                        match = bet3000.soccer_matches(sports, country, bookie, competition, url)
                    elif bookie == 'bet365':
                        match = bet365.soccer_matches(sports, country, bookie, competition, url)
                    elif bookie == 'sport888':
                        match = sport888.soccer_matches(sports, country, bookie, competition, url)
                    elif bookie == 'betway':
                        match = betway.soccer_matches(sports, country, bookie, competition, url)
                    elif bookie == 'betmaster':
                        match = betmaster.soccer_matches(sports, country, bookie, competition, url)
                    elif bookie == 'fdj':
                        match = fdj.soccer_matches(sports, country, bookie, competition, url)
                    elif bookie == 'pokerstars':
                        match = pokerstars.soccer_matches(sports, country, bookie, competition, url)
                    elif bookie == 'unibet':
                        match = unibet.soccer_matches(sports, country, bookie, competition, url)
                    elif bookie == 'betclic':
                        match = betclic.soccer_matches(sports, country, bookie, competition, url)
                    elif bookie == 'tiptorro':
                        match = tiptorro.soccer_matches(sports, country, bookie, competition, url)
                    elif bookie == 'tipico':
                        match = tipico.soccer_matches(sports, country, bookie, competition, url)
                    elif bookie == 'happybet':
                        match = happybet.soccer_matches(sports, country, bookie, competition, url)
                    elif bookie == 'sportwetten_de':
                        match = sportwetten_de.soccer_matches(sports, country, bookie, competition, url)
                    elif bookie == 'duelbits':
                        match = duelbits.soccer_matches(sports, country, bookie, competition, url)
                    elif bookie == 'stake':
                        match = stake.soccer_matches(sports, country, bookie, competition, url)
                    elif bookie == 'lsbet':
                        match = lsbet.soccer_matches(sports, country, bookie, competition, url)
                    elif bookie == 'cloudbet':
                        match = cloudbet.soccer_matches(sports, country, bookie, competition, url)
                    elif bookie == 'caesars':
                        match = caesars.soccer_matches(sports, country, bookie, competition, url)
                    elif bookie == 'draftkings':
                        match = draftkings.soccer_matches(sports, country, bookie, competition, url)
                    elif bookie == 'fanduel':
                        match = fanduel.soccer_matches(sports, country, bookie, competition, url)
                    elif bookie == 'virginbet':
                        virginbet_match = virginbet.soccer_matches(sports, country, bookie, competition, url)
                        skybet_match = skybet.soccer_matches(virginbet_match)
                        livescorebet_match = livescorebet.soccer_matches(virginbet_match)
                        match = virginbet_match._append(skybet_match, ignore_index=True)
                        match = match._append(livescorebet_match, ignore_index=True)
                    elif bookie == 'pinnacle':
                        pinnacle_match = pinnacle_api.soccer_matches(sports, country, bookie, competition, url)
                        piwi247_match = piwi247.soccer_matches(pinnacle_match)
                        ps3838_match = ps3838.soccer_matches(pinnacle_match)
                        asianodds_match = asianodds.soccer_matches(pinnacle_match)
                        match = pinnacle_match._append(piwi247_match, ignore_index=True)
                        match = match._append(ps3838_match, ignore_index=True)
                        match = match._append(asianodds_match, ignore_index=True)
                    elif bookie == 'chillybets':
                        chillybets_match = chillybets.soccer_matches(sports, country, bookie, competition, url)
                        onexbet_match = onexbet.soccer_matches(chillybets_match)
                        gastonred_match = gastonred.soccer_matches(chillybets_match)
                        librabet_match = librabet.soccer_matches(chillybets_match)
                        campeonbet_match = campeonbet.soccer_matches(chillybets_match)
                        alphawin_match = alphawin.soccer_matches(chillybets_match)
                        kto_match = kto.soccer_matches(chillybets_match)
                        fezbet_match = fezbet.soccer_matches(chillybets_match)
                        powbet_match = powbet.soccer_matches(chillybets_match)
                        sportaza_match = sportaza.soccer_matches(chillybets_match)
                        evobet_match = evobet.soccer_matches(chillybets_match)
                        quickwin_match = quickwin.soccer_matches(chillybets_match)
                        bankonbet_match = bankonbet.soccer_matches(chillybets_match)
                        sgcasino_match = sgcasino.soccer_matches(chillybets_match)
                        betsamigo_match = betsamigo.soccer_matches(chillybets_match)
                        greatwin_match = greatwin.soccer_matches(chillybets_match)
                        playzilla_match = playzilla.soccer_matches(chillybets_match)
                        nucleonbet_match = nucleonbet.soccer_matches(chillybets_match)
                        rabona_match = rabona.soccer_matches(chillybets_match)
                        betstro_match = betstro.soccer_matches(chillybets_match)
                        wazamba_match = wazamba.soccer_matches(chillybets_match)
                        lottoland_match = lottoland.soccer_matches(chillybets_match)
                        magicalvegas_match = magicalvegas.soccer_matches(chillybets_match)
                        match = chillybets_match._append(onexbet_match, ignore_index=True)
                        match = match._append(gastonred_match, ignore_index=True)
                        match = match._append(librabet_match, ignore_index=True)
                        match = match._append(campeonbet_match, ignore_index=True)
                        match = match._append(alphawin_match, ignore_index=True)
                        match = match._append(kto_match, ignore_index=True)
                        match = match._append(fezbet_match, ignore_index=True)
                        match = match._append(powbet_match, ignore_index=True)
                        match = match._append(sportaza_match, ignore_index=True)
                        match = match._append(evobet_match, ignore_index=True)
                        match = match._append(quickwin_match, ignore_index=True)
                        match = match._append(bankonbet_match, ignore_index=True)
                        match = match._append(sgcasino_match, ignore_index=True)
                        match = match._append(betsamigo_match, ignore_index=True)
                        match = match._append(greatwin_match, ignore_index=True)
                        match = match._append(playzilla_match, ignore_index=True)
                        match = match._append(nucleonbet_match, ignore_index=True)
                        match = match._append(rabona_match, ignore_index=True)
                        match = match._append(betstro_match, ignore_index=True)
                        match = match._append(wazamba_match, ignore_index=True)
                        match = match._append(lottoland_match, ignore_index=True)
                        match = match._append(magicalvegas_match, ignore_index=True)
                    elif bookie == 'ladbrokes':
                        ladbrokes_match = ladbrokes.soccer_matches(sports, country, bookie, competition, url)
                        bwin_match = bwin.soccer_matches(ladbrokes_match)
                        gamebookers_match = gamebookers.soccer_matches(ladbrokes_match)
                        sportingbet_match = sportingbet.soccer_matches(ladbrokes_match)
                        oddset_match = oddset.soccer_matches(ladbrokes_match)
                        bpremium_match = bpremium.soccer_matches(ladbrokes_match)
                        vistabet_match = vistabet.soccer_matches(ladbrokes_match)
                        betmgm_match = betmgm.soccer_matches(ladbrokes_match)
                        match = ladbrokes_match._append(bwin_match, ignore_index=True)
                        match = match._append(gamebookers_match, ignore_index=True)
                        match = match._append(sportingbet_match, ignore_index=True)
                        match = match._append(oddset_match, ignore_index=True)
                        match = match._append(bpremium_match, ignore_index=True)
                        match = match._append(vistabet_match, ignore_index=True)
                        match = match._append(betmgm_match, ignore_index=True)
                    elif bookie == 'mybet':
                        mybet_match = mybet_api.soccer_matches(sports, country, bookie, competition, url)
                        expekt_match = expekt.soccer_matches(mybet_match)
                        casumo_match = casumo.soccer_matches(mybet_match)
                        leovegas_match = leovegas.soccer_matches(mybet_match)
                        threetwored2_match = threetwored2.soccer_matches(mybet_match)
                        betplay_match = betplay.soccer_matches(mybet_match)
                        betuk_match = betuk.soccer_matches(mybet_match)
                        match = mybet_match._append(expekt_match, ignore_index=True)
                        match = match._append(casumo_match, ignore_index=True)
                        match = match._append(leovegas_match, ignore_index=True)
                        match = match._append(threetwored2_match, ignore_index=True)
                        match = match._append(betplay_match, ignore_index=True)
                        match = match._append(betuk_match, ignore_index=True)
                    elif bookie == 'mystake':
                        mystake_match = mystake.soccer_matches(sports, country, bookie, competition, url)
                        freshbet_match = freshbet.soccer_matches(mystake_match)
                        goldenbet_match = goldenbet.soccer_matches(mystake_match)
                        jackbit_match = jackbit.soccer_matches(mystake_match)
                        threeonebet_match = threeonebet.soccer_matches(mystake_match)
                        match = mystake_match._append(freshbet_match, ignore_index=True)
                        match = match._append(goldenbet_match, ignore_index=True)
                        match = match._append(jackbit_match, ignore_index=True)
                        match = match._append(threeonebet_match, ignore_index=True)
                    elif bookie == 'vave':
                        vave_match = vave.soccer_matches(sports, country, bookie, competition, url)
                        twozerobet_match = twozerobet.soccer_matches(vave_match)
                        ivibet_match = ivibet.soccer_matches(vave_match)
                        match = vave_match._append(twozerobet_match, ignore_index=True)
                        match = match._append(ivibet_match, ignore_index=True)
                    elif bookie == '22bet':
                        twotwobet_match = twotwobet.soccer_matches(sports, country, bookie, competition, url)
                        paripesa_match = paripesa.soccer_matches(twotwobet_match)
                        megapari_match = megapari.soccer_matches(twotwobet_match)
                        onexbit_match = onexbit.soccer_matches(twotwobet_match)
                        match = twotwobet_match._append(paripesa_match, ignore_index=True)
                        match = match._append(megapari_match, ignore_index=True)
                        match = match._append(onexbit_match, ignore_index=True)
                    elif bookie == 'n1bet':
                        n1bet_match = n1bet.soccer_matches(sports, country, bookie, competition, url)
                        bambet_match = bambet.soccer_matches(n1bet_match)
                        cobrabet_match = cobrabet.soccer_matches(n1bet_match)
                        rocketplay_match = rocketplay.soccer_matches(n1bet_match)
                        qbet_match = qbet.soccer_matches(n1bet_match)
                        winz_match = winz.soccer_matches(n1bet_match)
                        betibet_match = betibet.soccer_matches(n1bet_match)
                        winning_match = winning.soccer_matches(n1bet_match)
                        zotabet_match = zotabet.soccer_matches(n1bet_match)
                        match = n1bet_match._append(bambet_match, ignore_index=True)
                        match = match._append(cobrabet_match, ignore_index=True)
                        match = match._append(rocketplay_match, ignore_index=True)
                        match = match._append(qbet_match, ignore_index=True)
                        match = match._append(winz_match, ignore_index=True)
                        match = match._append(betibet_match, ignore_index=True)
                        match = match._append(winning_match, ignore_index=True)
                        match = match._append(zotabet_match, ignore_index=True)
                    elif bookie == 'dafabet':
                        dafabet_match = dafabet.soccer_matches(sports, country, bookie, competition, url)
                        nextbet_match = nextbet.soccer_matches(dafabet_match)
                        match = dafabet_match._append(nextbet_match, ignore_index=True)
                    elif bookie == 'betvictor':
                        betvictor_match = betvictor.soccer_matches(sports, country, bookie, competition, url)
                        bildbet_match = bildbet.soccer_matches(betvictor_match)
                        parimatch_match = parimatch.soccer_matches(betvictor_match)
                        match = betvictor_match._append(bildbet_match, ignore_index=True)
                        match = match._append(parimatch_match, ignore_index=True)
                    elif bookie == 'bcgame':
                        bcgame_match = bcgame.soccer_matches(sports, country, bookie, competition, url)
                        roobet_match = roobet.soccer_matches(bcgame_match)
                        yonibet_match = yonibet.soccer_matches(bcgame_match)
                        casinozer_match = casinozer.soccer_matches(bcgame_match)
                        solcasino_match = solcasino.soccer_matches(bcgame_match)
                        rollbit_match = rollbit.soccer_matches(bcgame_match)
                        nearcasino_match = nearcasino.soccer_matches(bcgame_match)
                        gamblingapes_match = gamblingapes.soccer_matches(bcgame_match)
                        joycasino_match = joycasino.soccer_matches(bcgame_match)
                        moonbet_match = moonbet.soccer_matches(bcgame_match)
                        bluechip_match = bluechip.soccer_matches(bcgame_match)
                        rajbets_match = rajbets.soccer_matches(bcgame_match)
                        betfury_match = betfury.soccer_matches(bcgame_match)
                        owlgames_match = owlgames.soccer_matches(bcgame_match)
                        terracasino_match = terracasino.soccer_matches(bcgame_match)
                        match = bcgame_match._append(roobet_match, ignore_index=True)
                        match = match._append(yonibet_match, ignore_index=True)
                        match = match._append(casinozer_match, ignore_index=True)
                        match = match._append(solcasino_match, ignore_index=True)
                        match = match._append(rollbit_match, ignore_index=True)
                        match = match._append(nearcasino_match, ignore_index=True)
                        match = match._append(gamblingapes_match, ignore_index=True)
                        match = match._append(joycasino_match, ignore_index=True)
                        match = match._append(moonbet_match, ignore_index=True)
                        match = match._append(bluechip_match, ignore_index=True)
                        match = match._append(rajbets_match, ignore_index=True)
                        match = match._append(betfury_match, ignore_index=True)
                        match = match._append(owlgames_match, ignore_index=True)
                        match = match._append(terracasino_match, ignore_index=True)


                    match['sport'] = sports
                    match['country'] = country
                    match['competition'] = competition
                    matches = matches._append(match, ignore_index=True)

                elif sports == 'basketball':
                    if bookie == 'betano':
                        match = betano.basketball_matches(sports, country, bookie, competition, url)
                    elif bookie == 'bet365':
                        match = bet365.basketball_matches(sports, country, bookie, competition, url)
                    elif bookie == 'sport888':
                        match = sport888.basketball_matches(sports, country, bookie, competition, url)
                    elif bookie == 'ladbrokes':
                        ladbrokes_match = ladbrokes.basketball_matches(sports, country, bookie, competition, url)
                        bwin_match = bwin.basketball_matches(ladbrokes_match)
                        gamebookers_match = gamebookers.basketball_matches(ladbrokes_match)
                        sportingbet_match = sportingbet.basketball_matches(ladbrokes_match)
                        oddset_match = oddset.basketball_matches(ladbrokes_match)
                        bpremium_match = bpremium.basketball_matches(ladbrokes_match)
                        vistabet_match = vistabet.basketball_matches(ladbrokes_match)
                        betmgm_match = betmgm.basketball_matches(ladbrokes_match)
                        match = ladbrokes_match._append(bwin_match, ignore_index=True)
                        match = match._append(gamebookers_match, ignore_index=True)
                        match = match._append(sportingbet_match, ignore_index=True)
                        match = match._append(oddset_match, ignore_index=True)
                        match = match._append(bpremium_match, ignore_index=True)
                        match = match._append(vistabet_match, ignore_index=True)
                        match = match._append(betmgm_match, ignore_index=True)
                    elif bookie == 'mybet':
                        mybet_match = mybet_api.basketball_matches(sports, country, bookie, competition, url)
                        expekt_match = expekt.basketball_matches(mybet_match)
                        casumo_match = casumo.basketball_matches(mybet_match)
                        leovegas_match = leovegas.basketball_matches(mybet_match)
                        threetwored2_match = threetwored2.basketball_matches(mybet_match)
                        betplay_match = betplay.basketball_matches(mybet_match)
                        betuk_match = betuk.basketball_matches(mybet_match)
                        match = mybet_match._append(expekt_match, ignore_index=True)
                        match = match._append(casumo_match, ignore_index=True)
                        match = match._append(leovegas_match, ignore_index=True)
                        match = match._append(threetwored2_match, ignore_index=True)
                        match = match._append(betplay_match, ignore_index=True)
                        match = match._append(betuk_match, ignore_index=True)

                    match['sport'] = sports
                    match['country'] = country
                    match['competition'] = competition
                    matches = matches._append(match, ignore_index=True)
            except:
                a = 1
        return matches


# for all sports
def getmatches_chrome(chrome_competitions):

    if len(chrome_competitions) > 0:
        tabs_count_games = 9
        chrome_competitions = chrome_competitions._append(chrome_competitions.head(1), ignore_index=True)
        sports = chrome_competitions['sports'].values[0]
        country = chrome_competitions['country'].values[0]
        competition = chrome_competitions['competition'][0]
        matches = pd.DataFrame()

        chrome_options = Options()
        chrome_options.add_argument("--incognito")
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('start-maximized')
        chrome_options.add_argument('disable-infobars')
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--headless=new")
        driver = webdriver.Chrome(options=chrome_options, executable_path='/usr/bin/chromedriver')
        #driver = webdriver.Firefox(options=chrome_options)
        for t in range(1, tabs_count_games):
            driver.execute_script("window.open('about:blank', 'tab{}');".format(t + 1))
        abbruch = int(len(chrome_competitions['sports']) - tabs_count_games / tabs_count_games + 1)
        for runs in range(0, abbruch):  # anzahl der durchläufe in zweier schritten durch die leagues csv
            length = []
            for t in range(0, tabs_count_games):
                length.append(runs * tabs_count_games + t + 1)
            this_session_leagues = chrome_competitions.loc[chrome_competitions.index.isin(length)].reset_index(drop=True)
            for l in range(0, len(this_session_leagues)):
                driver.switch_to.window(driver.window_handles[l])
                try:
                    url = this_session_leagues['url'][l]
                    driver.get(url)
                    print('in progress:', url)
                except:
                    print('failed!')
                time.sleep(0.2)

            for l in range(0, len(this_session_leagues)):
                driver.switch_to.window(driver.window_handles[l])
                one_league = this_session_leagues.loc[this_session_leagues.index == l].reset_index(drop=True)
                try:
                    bookie = one_league['bookie'].values[0]
                    url = one_league['url'][0]
                    match = ''
                    if sports == 'soccer':
                        if bookie == 'interwetten':
                            match = interwetten.soccer_matches(driver, sports, country, bookie, competition, url)
                        elif bookie == 'admiralbet':
                            match = admiralbet.soccer_matches(driver, sports, country, bookie, competition, url)
                        elif bookie == 'merkur_sports':
                            match = merkur_sports.soccer_matches(driver, sports, country, bookie, competition, url)
                        elif bookie == 'betathome':
                            match = betathome.soccer_matches(driver, sports, country, bookie, competition, url)
                        elif bookie == 'daznbet':
                            match = daznbet.soccer_matches(driver, sports, country, bookie, competition, url)
                        elif bookie == 'neobet':
                            match = neobet.soccer_matches(driver, sports, country, bookie, competition, url)
                        elif bookie == 'winamax':
                            match = winamax.soccer_matches(driver, sports, country, bookie, competition, url)
                        elif bookie == 'tipwin':
                            match = tipwin.soccer_matches(driver, sports, country, bookie, competition, url)
                        elif bookie == 'genybet':
                            match = genybet.soccer_matches(driver, sports, country, bookie, competition, url)
                        elif bookie == 'cbet':
                            match = cbet.soccer_matches(driver, sports, country, bookie, competition, url)
                        elif bookie == 'joabet':
                            match = joabet.soccer_matches(driver, sports, country, bookie, competition, url)
                        elif bookie == 'pmu':
                            match = pmu.soccer_matches(driver, sports, country, bookie, competition, url)
                        elif bookie == 'netbet':
                            match = netbet.soccer_matches(driver, sports, country, bookie, competition, url)
                        elif bookie == 'wolfbet':
                            match = wolfbet.soccer_matches(driver, sports, country, bookie, competition, url)
                        elif bookie == 'zebet':
                            match = zebet.soccer_matches(driver, sports, country, bookie, competition, url)
                        elif bookie == 'leonbet':
                            match = leonbet.soccer_matches(driver, sports, country, bookie, competition, url)
                        elif bookie == 'mobilebet':
                            mobilebet_match = mobilebet.soccer_matches(driver, sports, country, bookie, competition, url)
                            jet10_match = jets10.soccer_matches(mobilebet_match)
                            sultanbet_match = sultanbet.soccer_matches(mobilebet_match)
                            lilibet_match = lilibet.soccer_matches(mobilebet_match)
                            cricbaba_match = cricbaba.soccer_matches(mobilebet_match)
                            match = mobilebet_match._append(jet10_match, ignore_index=True)
                            match = match._append(sultanbet_match, ignore_index=True)
                            match = match._append(lilibet_match, ignore_index=True)
                            match = match._append(cricbaba_match, ignore_index=True)
                        elif bookie == 'dreambet':
                            dreambet_match = dreambet.soccer_matches(driver, sports, country, bookie, competition, url)
                            onebet_match = onebet.soccer_matches(dreambet_match)
                            betobet_match = betobet.soccer_matches(dreambet_match)
                            weltbet_match = weltbet.soccer_matches(dreambet_match)
                            olympusbet_match = olympusbet.soccer_matches(dreambet_match)
                            betrophy_match = betrophy.soccer_matches(dreambet_match)
                            dachbet_match = dachbet.soccer_matches(dreambet_match)
                            cashalot_match = cashalot.soccer_matches(dreambet_match)
                            match = dreambet_match._append(onebet_match, ignore_index=True)
                            match = match._append(betobet_match, ignore_index=True)
                            match = match._append(weltbet_match, ignore_index=True)
                            match = match._append(olympusbet_match, ignore_index=True)
                            match = match._append(betrophy_match, ignore_index=True)
                            match = match._append(dachbet_match, ignore_index=True)
                            match = match._append(cashalot_match, ignore_index=True)



                        match['sport'] = sports
                        match['country'] = country
                        match['competition'] = competition
                        matches = matches._append(match, ignore_index=True)
                        time.sleep(0.2)

                    elif sports == 'basketball':
                        print('We dont offer this sport yet.')
                except:
                    a = 1
        driver.close()
        return matches


# for all sports
def same_events_preprocessing(matches):

    if len(matches) > 2:
        matches['home_team'] = matches['home_team'].replace('VfL BOCHUM', 'VfL Bochum')
        matches['match'] = matches['home_team'] + ' vs ' + matches['away_team']
        matches['country'] = matches['country'].astype(str)
        matches['match'] = matches['match'].astype(str)
        matches['match'] = matches['match'].str.replace(' fc', '').replace('fc', '').replace(' FC', '').replace(
                'FC', '').replace(' u19', ' j-team ').replace(' u23', ' j-team ').replace(' u ', ' j-team ').replace(
                ' U19', ' j-team ').replace(' U20', ' j-team ').replace(' U21', ' j-team ').replace(' U23', ' j-team ').str.lower().replace(r'[0-9]', '', regex=True)

        texts_df = pd.DataFrame()
        texts_df['match2'] = list(set((matches['match'])))
        try:
            translated = GoogleTranslator(source='auto', target='en').translate_batch(texts_df['match2'].tolist())
            texts_df['match'] = translated
        except:
            texts_df['match'] = texts_df['match2']
        matches = pd.merge(matches, texts_df, on='match', how='left')
        del matches['match2']

        x = []
        for i in matches['match']:
            i = i.replace('\u00f6', 'ö').replace('\u00d6', 'ö').replace('\u00c4', 'ä').replace('\u00e4', 'ä')
            i = i.replace('\u00dc', 'ü').replace('\u00fc', 'ü').replace('\u00df', 'ß').replace('\u00e9', 'e')
            i = i.replace('\u00ee', 'i').replace('%C3%BC', 'ü').replace('%C3%B6', 'ö').replace('%C3%A4', 'ä').replace('\\', '')
            i = i.replace('-\\uc', 'e').replace(' hove albion', '').replace(' Hove Albion', '').replace(' &amp;', 'and')
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
            i = i.replace('calcio', '').replace('Calcio', '').replace('&amp', '')
            if 'j-team' in i:
                x.append('a')
            else:
                x.append(re.sub(r'[!@#$()"%^&*?:/.;~`0-9]', '', i.lower()))
        matches['match'] = x
        matches = matches.loc[matches['match'] != 'a'].reset_index(drop=True)
        matches = matches.loc[matches['match'] != 'nan'].reset_index(drop=True)
        matches = matches.loc[matches['match'] != 'home vs away'].reset_index(drop=True)
        return matches


# for all sports
def same_events(matches, sport, country, competition):

    unique_sports = matches['sport'].unique()
    unique_countries = matches['country'].unique()
    print(sorted(matches['bookie'].unique().tolist()))
    print(len(matches['bookie'].unique().tolist()))

    df_sorted = pd.DataFrame()
    for i in unique_sports:
        for j in unique_countries:
            df = matches.loc[(matches['sport'] == i) & (matches['country'] == j)].reset_index(drop=True)
            print(len(df))
            df = df.drop_duplicates(subset=['bookie', 'match']).reset_index(drop=True)

            betano = len(df.loc[df['bookie'] == 'betano'])  # ladbrokes auch möglich
            cloudbet = len(df.loc[df['bookie'] == 'cloudbet'])
            pinnacle = len(df.loc[df['bookie'] == 'pinnacle'])
            if max(betano, cloudbet, pinnacle) == betano:
                perfect_names = df.loc[df['bookie'] == 'betano'].reset_index(drop=True)
            elif max(betano, cloudbet, pinnacle) == pinnacle:
                perfect_names = df.loc[df['bookie'] == 'pinnacle'].reset_index(drop=True)
            elif max(betano, cloudbet, pinnacle) == cloudbet:
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
                if time_difference.days != 0 or time_difference.seconds / 60 / 60 > 2:
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
                            df_matched['id'] = 'id' + str(games)

                            belonging_ids = []
                            for anzahl in range(0, len(df_matched['match'])):
                                if 'vs' in str(df_matched['match'][anzahl]):
                                    unkown_home_team = df_matched['match'][anzahl].split('vs')[0]
                                    unkown_away_team = df_matched['match'][anzahl].split('vs')[1]

                                    seq_homes = difflib.SequenceMatcher(None, home_team, unkown_home_team).ratio() * 100
                                    hardcountries = ['argentina', 'brazil', 'world', 'greece', 'romania', 'euro']
                                    semicountries = ['usa', 'denmark', 'norway', 'czechia', 'spain', 'italy', 'france']
                                    if j in hardcountries:
                                        if seq_homes > 66:
                                            seq_aways = difflib.SequenceMatcher(None, away_team, unkown_away_team).ratio() * 100
                                            if seq_aways > 66:
                                                belonging_ids.append(anzahl)
                                    elif j in semicountries:
                                        if seq_homes > 64:
                                            seq_aways = difflib.SequenceMatcher(None, away_team, unkown_away_team).ratio() * 100
                                            if seq_aways > 64:
                                                belonging_ids.append(anzahl)
                                    else:
                                        if seq_homes > 62.5:
                                            seq_aways = difflib.SequenceMatcher(None, away_team, unkown_away_team).ratio() * 100
                                            if seq_aways > 62.5:
                                                belonging_ids.append(anzahl)

                            df_one_game = df_matched.loc[df_matched.index.isin(belonging_ids)].reset_index(drop=True)
                            df_one_game['match'] = str(game1)
                            df_one_game['date'].fillna(matchdate, inplace=True)
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
                            print(sorted([bookmaker for bookmaker in soccer_bookmakers() if bookmaker not in df_one_game['bookie'].unique().tolist()]))

                            df_sorted = df_sorted._append(df_one_game, ignore_index=True)

    df_sorted.drop_duplicates(subset=['bookie', 'match'], keep='last', inplace=True)
    df_sorted.to_csv(f'csv_data/{sport}/matched4{country}_{competition}.csv', index=False)
    return df_sorted


# for all sports
def preprocessing_for_return(same_matches, match_urls_key):
    matched = pd.DataFrame()
    del same_matches['competition_url'], same_matches['match_api']
    if 'id' in same_matches.columns:
        for i in same_matches['id'].unique().tolist():
            one_match = same_matches.loc[same_matches['id'] == i].sort_values(by='bookie', ascending=True).reset_index(drop=True)
            bookies = str(sorted([one_match['bookie'].unique().tolist()])).replace('[', '').replace(']', '')
            match_urls = str(sorted([one_match['match_url'].unique().tolist()])).replace('[', '').replace(']', '')
            one_match = one_match.drop_duplicates(subset='id')
            one_match['bookie'] = bookies
            one_match['match_url'] = match_urls
            matched = matched._append(one_match, ignore_index=True)
    else:
        ids = []
        matched = same_matches
        for m in range(0, len(matched)):
            ids.append(str('id') + str(m))
        matched['id'] = ids
    matched = matched.rename(columns={'id': 'matchid'})
    if match_urls_key != 'true':
        del matched['match_url']
    return matched


# for all sports, ('/api/v2/odds')    # CHECK FOR LIVE!! dann nicht in pregame_soccer_odds !
def getodds(sport, country, competition, matchid, bookmakers):
    if ',' in bookmakers:
        bookmakers = bookmakers.split(',')
    else:
        bookmakers = [bookmakers]
    matches = pd.read_csv(f'csv_data/{sport}/matched4{country}_{competition}.csv')

    bookmakers_cache = bookmakers
    bookmakers = [element.replace('bwin', 'ladbrokes').replace('sportingbet', 'ladbrokes').replace('oddset', 'ladbrokes').replace('gamebookers', 'ladbrokes') for element in bookmakers]
    bookmakers = [element.replace('bpremium', 'ladbrokes').replace('vistabet', 'ladbrokes').replace('betmgm', 'ladbrokes') for element in bookmakers]
    bookmakers = [element.replace('skybet', 'virginbet').replace('livescorebet', 'virginbet').replace('piwi247', 'pinnacle') for element in bookmakers]
    bookmakers = [element.replace('ps3838', 'pinnacle').replace('asianodds', 'pinnacle').replace('1xbet', 'chillybets') for element in bookmakers]
    bookmakers = [element.replace('gastonred', 'chillybets').replace('librabet', 'chillybets').replace('campeonbet', 'chillybets') for element in bookmakers]
    bookmakers = [element.replace('alphawin', 'chillybets').replace('kto', 'chillybets').replace('fezbet', 'chillybets') for element in bookmakers]
    bookmakers = [element.replace('powbet', 'chillybets').replace('sportaza', 'chillybets').replace('evobet', 'chillybets') for element in bookmakers]
    bookmakers = [element.replace('quickwin', 'chillybets').replace('bankonbet', 'chillybets').replace('sgcasino', 'chillybets') for element in bookmakers]
    bookmakers = [element.replace('betsamigo', 'chillybets').replace('greatwin', 'chillybets').replace('playzilla', 'chillybets') for element in bookmakers]
    bookmakers = [element.replace('nucleonbet', 'chillybets').replace('rabona', 'chillybets').replace('betstro', 'chillybets') for element in bookmakers]
    bookmakers = [element.replace('wazamba', 'chillybets').replace('lottoland', 'chillybets').replace('magicalvegas', 'chillybets') for element in bookmakers]
    bookmakers = [element.replace('expekt', 'mybet').replace('casumo', 'mybet').replace('leovegas', 'mybet') for element in bookmakers]
    bookmakers = [element.replace('32red', 'mybet').replace('betplay', 'mybet').replace('betuk', 'mybet') for element in bookmakers]
    bookmakers = [element.replace('freshbet', 'mystake').replace('goldenbet', 'mystake').replace('jackbit', 'mystake') for element in bookmakers]
    bookmakers = [element.replace('31bet', 'mystake').replace('20bet', 'vave').replace('ivibet', 'vave') for element in bookmakers]
    bookmakers = [element.replace('paripesa', '22bet').replace('megapari', '22bet').replace('1xbit', '22bet') for element in bookmakers]
    bookmakers = [element.replace('bambet', 'n1bet').replace('cobrabet', 'n1bet').replace('rocketplay', 'n1bet') for element in bookmakers]
    bookmakers = [element.replace('qbet', 'n1bet').replace('winz', 'n1bet').replace('betibet', 'n1bet') for element in bookmakers]
    bookmakers = [element.replace('winning', 'n1bet').replace('zotabet', 'n1bet').replace('nextbet', 'dafabet') for element in bookmakers]
    bookmakers = [element.replace('bildbet', 'betvictor').replace('parimatch', 'betvictor').replace('roobet', 'bcgame') for element in bookmakers]
    bookmakers = [element.replace('solcasino', 'bcgame').replace('rollbit', 'bcgame').replace('nearcasino', 'bcgame') for element in bookmakers]
    bookmakers = [element.replace('gamblingapes', 'bcgame').replace('joycasino', 'bcgame').replace('moonbet', 'bcgame') for element in bookmakers]
    bookmakers = [element.replace('bluechip', 'bcgame').replace('rajbets', 'bcgame').replace('betfury', 'bcgame') for element in bookmakers]
    bookmakers = [element.replace('owlgames', 'bcgame').replace('terracasino', 'bcgame').replace('yonibet', 'bcgame') for element in bookmakers]
    bookmakers = [element.replace('casinozer', 'bcgame').replace('n1bet', 'n1bed').replace('1bet', 'dreambet').replace('n1bed', 'n1bet').replace('betobet', 'dreambet') for element in bookmakers]
    bookmakers = [element.replace('weltbet', 'dreambet').replace('olympusbet', 'dreambet').replace('betrophy', 'dreambet') for element in bookmakers]
    bookmakers = [element.replace('dachbet', 'dreambet').replace('cashalot', 'dreambet').replace('jet10', 'mobilebet') for element in bookmakers]
    bookmakers = [element.replace('sultanbet', 'mobilebet').replace('lilibet', 'mobilebet').replace('cricbaba', 'mobilebet') for element in bookmakers]
    match = matches.loc[(matches['id'] == matchid) & (matches['bookie'].isin(bookmakers))].reset_index(drop=True)
    chrome_bookies = ['interwetten', 'admiralbet', 'merkur_sports', 'winamax', 'neobet', 'joabet', 'pmu', 'zebet',
                      'dreambet', 'mobilebet', 'tipwin', 'daznbet', 'betathome']
    print(match)
    api_match = match.loc[~match['bookie'].isin(chrome_bookies)].reset_index(drop=True)
    chrome_match = match.loc[match['bookie'].isin(chrome_bookies)].reset_index(drop=True)
    match_api = getodds_api(api_match)
    match_chrome = getodds_chrome(chrome_match)

    odds = match_api._append(match_chrome, ignore_index=True)
    odds = odds.loc[odds['bookie'].isin(bookmakers_cache)].reset_index(drop=True)
    odds = afterprocessing(odds)

    # SUREBETS ? VALUEBETS ?
    surebets = getsurebets_soccer(odds)
    odds = odds.transpose().fillna(value='').to_dict()
    odds = round_floats(odds, 3)
    return odds


# for all sports
def getodds_api(api_match):
    header = ['match_url', 'match_api', 'date', 'time', 'home_team', 'away_team', 'bookie',
              'sport', 'country', 'competition', 'match', 'id']
    odds = pd.DataFrame(columns=header)
    api_match = api_match.drop_duplicates(subset='match_api').reset_index(drop=True)
    if len(api_match) > 0:
        sports = api_match['sport'].values[0]
        country = api_match['country'].values[0]
        competition = api_match['competition'][0]
        matchname = api_match['match'][0]
        ids = api_match['id'][0]
        date = api_match['date'][0]
        times = api_match['time'][0]
        home_team = api_match['home_team'][0]
        away_team = api_match['away_team'][0]
        for k in range(0, len(api_match)):
            url = api_match['match_api'].values[k]
            match_url = api_match['match_url'].values[k]
            print('in progress:', url)
            bookie = api_match['bookie'].values[k]
            try:
                match = ''
                if sports == 'soccer':
                    if bookie in ['virginbet', 'skybet', 'livescorebet']:
                        virginbet_match = virginbet.soccer_odds(sports, country, bookie, competition, url, match_url)
                        skybet_match = skybet.soccer_odds(virginbet_match)
                        livescorebet_match = livescorebet.soccer_odds(virginbet_match)
                        match = virginbet_match._append(skybet_match, ignore_index=True)
                        match = match._append(livescorebet_match, ignore_index=True)
                    elif bookie in ['betano']:
                        match = betano.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['bet365']:
                        match = bet365.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['betway']:
                        match = betway.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['wettarena']:
                        match = wettarena.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['bet3000']:
                        match = bet3000.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['happybet']:
                        match = happybet.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['sport888']:
                        match = sport888.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['betmaster']:
                        match = betmaster.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['netbet']:
                        match = netbet.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['fdj']:
                        match = fdj.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['genybet']:
                        match = genybet.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['pokerstars']:
                        match = pokerstars.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['unibet']:
                        match = unibet.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['betclic']:
                        match = betclic.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['tiptorro']:
                        match = tiptorro.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['tipico']:
                        match = tipico.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['sportwetten_de']:
                        match = sportwetten_de.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['duelbits']:
                        match = duelbits.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['stake']:
                        match = stake.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['lsbet']:
                        match = lsbet.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['leonbet']:
                        match = leonbet.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['cloudbet']:
                        match = cloudbet.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['caesars']:
                        match = caesars.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['draftkings']:
                        match = draftkings.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['fanduel']:
                        match = fanduel.soccer_odds(sports, country, bookie, competition, url, match_url)
                    elif bookie in ['ladbrokes', 'bwin', 'gamebookers', 'sportingbet', 'oddset', 'bpremium', 'vistabet', 'betmgm']:
                        ladbrokes_match = ladbrokes.soccer_odds(sports, country, bookie, competition, url, match_url)
                        bwin_match = bwin.soccer_odds(ladbrokes_match)
                        gamebookers_match = gamebookers.soccer_odds(ladbrokes_match)
                        sportingbet_match = sportingbet.soccer_odds(ladbrokes_match)
                        oddset_match = oddset.soccer_odds(ladbrokes_match)
                        bpremium_match = bpremium.soccer_odds(ladbrokes_match)
                        vistabet_match = vistabet.soccer_odds(ladbrokes_match)
                        betmgm_match = betmgm.soccer_odds(ladbrokes_match)
                        match = ladbrokes_match._append(bwin_match, ignore_index=True)
                        match = match._append(gamebookers_match, ignore_index=True)
                        match = match._append(sportingbet_match, ignore_index=True)
                        match = match._append(oddset_match, ignore_index=True)
                        match = match._append(bpremium_match, ignore_index=True)
                        match = match._append(vistabet_match, ignore_index=True)
                        match = match._append(betmgm_match, ignore_index=True)
                    elif bookie in ['pinnacle', 'piwi247', 'ps3838', 'asianodds']:
                        pinnacle_match = pinnacle_api.soccer_odds(sports, country, bookie, competition, url, match_url)
                        piwi247_match = piwi247.soccer_odds(pinnacle_match)
                        ps3838_match = ps3838.soccer_odds(pinnacle_match)
                        asianodds_match = asianodds.soccer_odds(pinnacle_match)
                        match = pinnacle_match._append(piwi247_match, ignore_index=True)
                        match = match._append(ps3838_match, ignore_index=True)
                        match = match._append(asianodds_match, ignore_index=True)
                    elif bookie in ['chillybets', '1xbet', 'gastonred', 'librabet', 'campeonbet', 'alphawin', 'kto', 'fezbet', 'powbet', 'sportaza', 'evobet', 'quickwin', 'bankonbet', 'sgcasino', 'betsamigo', 'greatwin', 'playzilla', 'nucleonbet', 'rabona', 'betstro', 'wazamba', 'lottoland', 'magicalvegas']:
                        chillybets_match = chillybets.soccer_odds(sports, country, bookie, competition, url, match_url)
                        onexbet_match = onexbet.soccer_odds(chillybets_match)
                        gastonred_match = gastonred.soccer_odds(chillybets_match)
                        librabet_match = librabet.soccer_odds(chillybets_match)
                        campeonbet_match = campeonbet.soccer_odds(chillybets_match)
                        alphawin_match = alphawin.soccer_odds(chillybets_match)
                        kto_match = kto.soccer_odds(chillybets_match)
                        fezbet_match = fezbet.soccer_odds(chillybets_match)
                        powbet_match = powbet.soccer_odds(chillybets_match)
                        sportaza_match = sportaza.soccer_odds(chillybets_match)
                        evobet_match = evobet.soccer_odds(chillybets_match)
                        quickwin_match = quickwin.soccer_odds(chillybets_match)
                        bankonbet_match = bankonbet.soccer_odds(chillybets_match)
                        sgcasino_match = sgcasino.soccer_odds(chillybets_match)
                        betsamigo_match = betsamigo.soccer_odds(chillybets_match)
                        greatwin_match = greatwin.soccer_odds(chillybets_match)
                        playzilla_match = playzilla.soccer_odds(chillybets_match)
                        nucleonbet_match = nucleonbet.soccer_odds(chillybets_match)
                        rabona_match = rabona.soccer_odds(chillybets_match)
                        betstro_match = betstro.soccer_odds(chillybets_match)
                        wazamba_match = wazamba.soccer_odds(chillybets_match)
                        lottoland_match = lottoland.soccer_odds(chillybets_match)
                        magicalvegas_match = magicalvegas.soccer_odds(chillybets_match)
                        match = chillybets_match._append(onexbet_match, ignore_index=True)
                        match = match._append(gastonred_match, ignore_index=True)
                        match = match._append(librabet_match, ignore_index=True)
                        match = match._append(campeonbet_match, ignore_index=True)
                        match = match._append(alphawin_match, ignore_index=True)
                        match = match._append(kto_match, ignore_index=True)
                        match = match._append(fezbet_match, ignore_index=True)
                        match = match._append(powbet_match, ignore_index=True)
                        match = match._append(sportaza_match, ignore_index=True)
                        match = match._append(evobet_match, ignore_index=True)
                        match = match._append(quickwin_match, ignore_index=True)
                        match = match._append(bankonbet_match, ignore_index=True)
                        match = match._append(sgcasino_match, ignore_index=True)
                        match = match._append(betsamigo_match, ignore_index=True)
                        match = match._append(greatwin_match, ignore_index=True)
                        match = match._append(playzilla_match, ignore_index=True)
                        match = match._append(nucleonbet_match, ignore_index=True)
                        match = match._append(rabona_match, ignore_index=True)
                        match = match._append(betstro_match, ignore_index=True)
                        match = match._append(wazamba_match, ignore_index=True)
                        match = match._append(lottoland_match, ignore_index=True)
                        match = match._append(magicalvegas_match, ignore_index=True)
                    elif bookie in ['mybet', 'expekt', 'casumo', 'leovegas', '32red', 'betplay', 'betuk']:
                        mybet_match = mybet_api.soccer_odds(sports, country, bookie, competition, url, match_url)
                        expekt_match = expekt.soccer_odds(mybet_match)
                        casumo_match = casumo.soccer_odds(mybet_match)
                        leovegas_match = leovegas.soccer_odds(mybet_match)
                        threetwored2_match = threetwored2.soccer_odds(mybet_match)
                        betplay_match = betplay.soccer_odds(mybet_match)
                        betuk_match = betuk.soccer_odds(mybet_match)
                        match = mybet_match._append(expekt_match, ignore_index=True)
                        match = match._append(casumo_match, ignore_index=True)
                        match = match._append(leovegas_match, ignore_index=True)
                        match = match._append(threetwored2_match, ignore_index=True)
                        match = match._append(betplay_match, ignore_index=True)
                        match = match._append(betuk_match, ignore_index=True)
                    elif bookie in ['mystake', 'freshbet', 'goldenbet', 'jackbit', '31bet']:
                        mystake_match = mystake.soccer_odds(sports, country, bookie, competition, url, match_url)
                        freshbet_match = freshbet.soccer_odds(mystake_match)
                        goldenbet_match = goldenbet.soccer_odds(mystake_match)
                        jackbit_match = jackbit.soccer_odds(mystake_match)
                        threeonebet_match = threeonebet.soccer_odds(mystake_match)
                        match = mystake_match._append(freshbet_match, ignore_index=True)
                        match = match._append(goldenbet_match, ignore_index=True)
                        match = match._append(jackbit_match, ignore_index=True)
                        match = match._append(threeonebet_match, ignore_index=True)
                    elif bookie in ['vave', '20bet', 'ivibet']:
                        vave_match = vave.soccer_odds(sports, country, bookie, competition, url, match_url)
                        twozerobet_match = twozerobet.soccer_odds(vave_match)
                        ivibet_match = ivibet.soccer_odds(vave_match)
                        match = vave_match._append(twozerobet_match, ignore_index=True)
                        match = match._append(ivibet_match, ignore_index=True)
                    elif bookie in ['22bet', 'paripesa', 'megapari', '1xbit']:
                        twotwobet_match = twotwobet.soccer_odds(sports, country, bookie, competition, url, match_url)
                        paripesa_match = paripesa.soccer_odds(twotwobet_match)
                        megapari_match = megapari.soccer_odds(twotwobet_match)
                        onexbit_match = onexbit.soccer_odds(twotwobet_match)
                        match = twotwobet_match._append(paripesa_match, ignore_index=True)
                        match = match._append(megapari_match, ignore_index=True)
                        match = match._append(onexbit_match, ignore_index=True)
                    elif bookie in ['n1bet', 'bambet', 'cobrabet', 'rocketplay', 'qbet', 'winz', 'betibet', 'winning', 'zotabet']:
                        n1bet_match = n1bet.soccer_odds(sports, country, bookie, competition, url, match_url)
                        bambet_match = bambet.soccer_odds(n1bet_match)
                        cobrabet_match = cobrabet.soccer_odds(n1bet_match)
                        rocketplay_match = rocketplay.soccer_odds(n1bet_match)
                        qbet_match = qbet.soccer_odds(n1bet_match)
                        winz_match = winz.soccer_odds(n1bet_match)
                        betibet_match = betibet.soccer_odds(n1bet_match)
                        winning_match = winning.soccer_odds(n1bet_match)
                        zotabet_match = zotabet.soccer_odds(n1bet_match)
                        match = n1bet_match._append(bambet_match, ignore_index=True)
                        match = match._append(cobrabet_match, ignore_index=True)
                        match = match._append(rocketplay_match, ignore_index=True)
                        match = match._append(qbet_match, ignore_index=True)
                        match = match._append(winz_match, ignore_index=True)
                        match = match._append(betibet_match, ignore_index=True)
                        match = match._append(winning_match, ignore_index=True)
                        match = match._append(zotabet_match, ignore_index=True)
                    elif bookie in ['dafabet', 'nextbet']:
                        dafabet_match = dafabet.soccer_odds(sports, country, bookie, competition, url, match_url)
                        nextbet_match = nextbet.soccer_odds(dafabet_match)
                        match = dafabet_match._append(nextbet_match, ignore_index=True)
                    elif bookie in ['betvictor', 'bildbet', 'parimatch']:
                        betvictor_match = betvictor.soccer_odds(sports, country, bookie, competition, url, match_url)
                        bildbet_match = bildbet.soccer_odds(betvictor_match)
                        parimatch_match = parimatch.soccer_odds(betvictor_match)
                        match = betvictor_match._append(bildbet_match, ignore_index=True)
                        match = match._append(parimatch_match, ignore_index=True)
                    elif bookie in ['bcgame', 'roobet', 'solcasino', 'rollbit', 'nearcasino', 'gambilngapes', 'joycasino', 'moonbet', 'bluechip', 'rajbets', 'betfury', 'owlgames', 'terracasino', 'yonibet', 'casinozer']:
                        bcgame_match = bcgame.soccer_odds(sports, country, bookie, competition, url, match_url)
                        roobet_match = roobet.soccer_odds(bcgame_match)
                        yonibet_match = yonibet.soccer_odds(bcgame_match)
                        casinozer_match = casinozer.soccer_odds(bcgame_match)
                        solcasino_match = solcasino.soccer_odds(bcgame_match)
                        rollbit_match = rollbit.soccer_odds(bcgame_match)
                        nearcasino_match = nearcasino.soccer_odds(bcgame_match)
                        gamblingapes_match = gamblingapes.soccer_odds(bcgame_match)
                        joycasino_match = joycasino.soccer_odds(bcgame_match)
                        moonbet_match = moonbet.soccer_odds(bcgame_match)
                        bluechip_match = bluechip.soccer_odds(bcgame_match)
                        rajbets_match = rajbets.soccer_odds(bcgame_match)
                        betfury_match = betfury.soccer_odds(bcgame_match)
                        owlgames_match = owlgames.soccer_odds(bcgame_match)
                        terracasino_match = terracasino.soccer_odds(bcgame_match)
                        match = bcgame_match._append(roobet_match, ignore_index=True)
                        match = match._append(yonibet_match, ignore_index=True)
                        match = match._append(casinozer_match, ignore_index=True)
                        match = match._append(solcasino_match, ignore_index=True)
                        match = match._append(rollbit_match, ignore_index=True)
                        match = match._append(nearcasino_match, ignore_index=True)
                        match = match._append(gamblingapes_match, ignore_index=True)
                        match = match._append(joycasino_match, ignore_index=True)
                        match = match._append(moonbet_match, ignore_index=True)
                        match = match._append(bluechip_match, ignore_index=True)
                        match = match._append(rajbets_match, ignore_index=True)
                        match = match._append(betfury_match, ignore_index=True)
                        match = match._append(owlgames_match, ignore_index=True)
                        match = match._append(terracasino_match, ignore_index=True)

                    match['sport'] = sports
                    match['country'] = country
                    match['competition'] = competition
                    match['match'] = matchname
                    match['id'] = ids
                    match['date'] = date
                    match['time'] = times
                    match['home_team'] = home_team
                    match['away_team'] = away_team
                    odds = odds._append(match, ignore_index=True)

                elif sports == 'basketball':
                    print('We dont offer this sport yet.')
            except:
                a = 1
    return odds


# for all sports
def getodds_chrome(chrome_match):

    header = ['match_url', 'match_api', 'date', 'time', 'home_team', 'away_team', 'bookie',
              'sport', 'country', 'competition', 'match', 'id']
    odds = pd.DataFrame(columns=header)
    chrome_match = chrome_match.drop_duplicates(subset='match_api').reset_index(drop=True)
    if len(chrome_match) > 0:
        tabs_count_games = 8
        sports = chrome_match['sport'].values[0]
        country = chrome_match['country'].values[0]
        competition = chrome_match['competition'][0]
        matchname = chrome_match['match'][0]
        ids = chrome_match['id'][0]
        date = chrome_match['date'][0]
        times = chrome_match['time'][0]
        home_team = chrome_match['home_team'][0]
        away_team = chrome_match['away_team'][0]

        chrome_options = Options()
        chrome_options.add_argument("--incognito")
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('start-maximized')
        chrome_options.add_argument('disable-infobars')
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--headless=new")
        driver = webdriver.Chrome(options=chrome_options, executable_path='/usr/bin/chromedriver')
        #driver = webdriver.Firefox(options=chrome_options)
        for t in range(1, tabs_count_games):
            driver.execute_script("window.open('about:blank', 'tab{}');".format(t + 1))
        abbruch = int(len(chrome_match['sport']) - tabs_count_games / tabs_count_games + 1)
        for runs in range(0, abbruch):  # anzahl der durchläufe in zweier schritten durch die leagues csv
            length = []
            for t in range(0, tabs_count_games):
                length.append(runs * tabs_count_games + t + 1)
            this_session_leagues = chrome_match.loc[chrome_match.index.isin(length)].reset_index(drop=True)
            if len(chrome_match) <= tabs_count_games/2:
                this_session_leagues = this_session_leagues._append(chrome_match, ignore_index=True)
            for l in range(0, len(this_session_leagues)):
                driver.switch_to.window(driver.window_handles[l])
                try:
                    url = this_session_leagues['match_api'][l]
                    driver.get(url)
                    print('in progress:', url)
                except:
                    print('failed!')
                time.sleep(0.2)

            for l in range(0, len(this_session_leagues)):
                driver.switch_to.window(driver.window_handles[l])
                one_league = this_session_leagues.loc[this_session_leagues.index == l].reset_index(drop=True)
                try:
                    url = one_league['match_url'].values[l]
                    match_url = one_league['match_url'].values[l]
                    bookie = one_league['bookie'].values[l]
                    match = ''
                    if sports == 'soccer':
                        if bookie in ['interwetten']:
                            match = interwetten.soccer_odds(driver, sports, country, bookie, competition, url, match_url)
                        elif bookie in ['admiralbet']:
                            match = admiralbet.soccer_odds(driver, sports, country, bookie, competition, url, match_url)
                        elif bookie in ['merkur_sports']:
                            match = merkur_sports.soccer_odds(driver, sports, country, bookie, competition, url, match_url)
                        elif bookie in ['betathome']:
                            match = betathome.soccer_odds(driver, sports, country, bookie, competition, url, match_url)
                        elif bookie in ['tipwin']:
                            match = tipwin.soccer_odds(driver, sports, country, bookie, competition, url, match_url)
                        elif bookie in ['neobet']:
                            match = neobet.soccer_odds(driver, sports, country, bookie, competition, url, match_url)
                        elif bookie in ['winamax']:
                            match = winamax.soccer_odds(driver, sports, country, bookie, competition, url, match_url)
                        elif bookie in ['joabet']:
                            match = joabet.soccer_odds(driver, sports, country, bookie, competition, url, match_url)
                        elif bookie in ['pmu']:
                            match = pmu.soccer_odds(driver, sports, country, bookie, competition, url, match_url)
                        elif bookie in ['zebet']:
                            match = zebet.soccer_odds(driver, sports, country, bookie, competition, url, match_url)
                        elif bookie in ['dreambet', '1bet', 'betobet', 'weltbet', 'olympusbet', 'betrophy', 'dachbet', 'cashalot']:
                            dreambet_match = dreambet.soccer_odds(driver, sports, country, bookie, competition, url, match_url)
                            onebet_match = onebet.soccer_odds(dreambet_match)
                            betobet_match = betobet.soccer_odds(dreambet_match)
                            weltbet_match = weltbet.soccer_odds(dreambet_match)
                            olympusbet_match = olympusbet.soccer_odds(dreambet_match)
                            betrophy_match = betrophy.soccer_odds(dreambet_match)
                            dachbet_match = dachbet.soccer_odds(dreambet_match)
                            cashalot_match = cashalot.soccer_odds(dreambet_match)
                            match = dreambet_match._append(onebet_match, ignore_index=True)
                            match = match._append(betobet_match, ignore_index=True)
                            match = match._append(weltbet_match, ignore_index=True)
                            match = match._append(olympusbet_match, ignore_index=True)
                            match = match._append(betrophy_match, ignore_index=True)
                            match = match._append(dachbet_match, ignore_index=True)
                            match = match._append(cashalot_match, ignore_index=True)
                        elif bookie in ['mobilebet', 'jet10', 'sultanbet', 'lilibet', 'cricbaba']:
                            mobilebet_match = mobilebet.soccer_odds(driver, sports, country, bookie, competition, url, match_url)
                            jets10_match = jets10.soccer_odds(mobilebet_match)
                            sultanbet_match = sultanbet.soccer_odds(mobilebet_match)
                            lilibet_match = lilibet.soccer_odds(mobilebet_match)
                            cricbaba_match = cricbaba.soccer_odds(mobilebet_match)
                            match = mobilebet_match._append(jets10_match, ignore_index=True)
                            match = match._append(sultanbet_match, ignore_index=True)
                            match = match._append(lilibet_match, ignore_index=True)
                            match = match._append(cricbaba_match, ignore_index=True)

                        match['sport'] = sports
                        match['country'] = country
                        match['competition'] = competition
                        match['match'] = matchname
                        match['id'] = ids
                        match['date'] = date
                        match['time'] = times
                        match['home_team'] = home_team
                        match['away_team'] = away_team
                        odds = odds._append(match, ignore_index=True)

                    elif sports == 'basketball':
                        print('We dont offer this sport yet.')
                except:
                    a = 1
        driver.close()
    return odds


# for all sports
def afterprocessing(odds):
    columns_to_modify1 = odds.columns
    columns_to_modify = []
    for c in columns_to_modify1:
        if c not in ['match_url', 'bookie', 'scraped_date', 'sport', 'country', 'competition', 'match', 'id', 'date', 'time', 'home_team', 'away_team']:
            columns_to_modify.append(c)
    for column in columns_to_modify:
        odds[column] = odds[column].astype(str)
        odds[column] = odds[column].str.replace(',', '.')
        odds[column] = odds[column].str.replace('"', '')
        odds[column] = odds[column].str.replace(
            r'[!@#$()"%^*<>?:}\/{;~"` abcdefghijklmnopqrstuvwxyzéABCDEFGHIJKLMNOPQRSTUVWXYZ-]\\', '', regex=True)
        x = []
        for i in odds[column]:
            x.append(str(re.sub(r'[!@#$()/"%^*<>,?:}{;~` abcdefghijklmnopqrstuvwxyzéABCDEFGHIJKLMNOPQRSTUVWXYZ]', '', i.lower())))
        odds[column] = x
        odds[column] = odds[column].astype(str)
        odds[column] = odds[column].fillna('None')
        odds[column] = odds[column].str.replace('\\', '')
        odds[column] = odds[column].str.replace(',', '.')
        odds[column] = odds[column].str.replace('[', '')
        odds[column] = odds[column].str.replace(']', '')
        odds[column] = odds[column].replace('None', None)
        odds[column] = odds[column].str.replace('1.00', '')
        odds[column] = odds[column].str.replace("'", '')
        odds[column] = odds[column].replace('', None)
    odds[columns_to_modify] = odds[columns_to_modify].astype(float)
    for column in columns_to_modify:
        odds[column] = np.round(odds[column], 3)
    odds['match_url'] = odds['match_url'].str.replace('\/', '/')
    del odds['match_api']
    return odds


# for all sports
def round_floats(d, places):
    if isinstance(d, float):
        return round(d, places)
    elif isinstance(d, dict):
        return {k: round_floats(v, places) for k, v in d.items()}
    else:
        return d


# for soccer
def getsurebets_soccer(odds):
    this_bet = odds.copy()
    if len(this_bet) > 1:
        columns_to_modify1 = odds.columns
        reciprocal_columns = []
        numeric_columns = []
        for c in columns_to_modify1:
            if c not in ['match_url', 'bookie', 'scraped_date', 'sport', 'country', 'competition', 'match', 'id',
                         'date', 'time', 'home_team', 'away_team']:
                numeric_columns.append(c)
                reciprocal_columns.append(f'p_{c}')

        this_bet[reciprocal_columns] = 1 / this_bet[numeric_columns].astype(float)
        df_surebets = pd.DataFrame()

        # HOME, DRAW, AWAY
        for ii in range(0, 1):
            home = this_bet.loc[this_bet['home'] == max(this_bet['home'])].reset_index(drop=True)
            draw = this_bet.loc[this_bet['draw'] == max(this_bet['draw'])].reset_index(drop=True)
            away = this_bet.loc[this_bet['away'] == max(this_bet['away'])].reset_index(drop=True)

            columns_to_reset = []
            for c in columns_to_modify1:
                if c not in ['match_url', 'bookie', 'scraped_date', 'sport', 'country', 'competition', 'match', 'id',
                             'date', 'time', 'home_team', 'away_team', 'home', 'draw', 'away']:
                    columns_to_reset.append(c)
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
            columns_to_reset = ['home', 'draw', 'away', 'o45', 'u45', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15', 'u05',
                                'o05',
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
                    df_surebet['profit_in_%'] = round(
                        100 / (b_score_y['p_b_score_y'][0] + b_score_n['p_b_score_n'][0]) - 100, 2)
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
            columns_to_reset = ['home', 'draw', 'away', 'o45', 'u45', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15', 'u05',
                                'o05',
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
            columns_to_reset = ['home', 'draw', 'away', 'o45', 'u45', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15', 'u05',
                                'o05',
                                'b_score_y', 'b_score_n', 'first_g_1', 'first_g_X', 'first_g_2', 'hand03_1', 'hand03_X',
                                'hand03_2', 'hand02_1',
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

            columns_to_reset = ['home', 'draw', 'away', 'o45', 'u45', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15', 'u05',
                                'o05',
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

            columns_to_reset = ['home', 'draw', 'away', 'o45', 'u45', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15', 'u05',
                                'o05',
                                'b_score_y', 'b_score_n', 'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1',
                                'first_h_X', 'first_h_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1', 'hand02_X',
                                'hand02_2',
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

            columns_to_reset = ['home', 'draw', 'away', 'o45', 'u45', 'o35', 'u35', 'u25', 'o25', 'o15', 'u15', 'u05',
                                'o05',
                                'b_score_y', 'b_score_n', 'first_g_1', 'first_g_X', 'first_g_2', 'first_h_1',
                                'first_h_X', 'first_h_2', 'hand03_1', 'hand03_X', 'hand03_2', 'hand02_1', 'hand02_X',
                                'hand02_2',
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
            df_surebets['link'] = df_surebets['url']
            del df_surebets['country'], df_surebets['p_home'], df_surebets['p_draw'], df_surebets['p_away'], df_surebets[
                'p_b_score_y'], df_surebets['p_b_score_n'], df_surebets['p_o45'], df_surebets['p_u45'], df_surebets[
                'p_o35'], df_surebets['p_u35']
            del df_surebets['url'], df_surebets['p_o25'], df_surebets['p_u25'], df_surebets['p_o15'], df_surebets['p_u15'], \
            df_surebets['p_o05'], df_surebets['p_u05']
            del df_surebets['p_first_g_1'], df_surebets['p_first_g_X'], df_surebets['p_first_g_2'], df_surebets[
                'p_first_h_1'], df_surebets['p_first_h_X'], df_surebets['p_first_h_2']
            del df_surebets['p_hand30_1'], df_surebets['p_hand30_X'], df_surebets['p_hand30_2'], df_surebets['p_hand20_1'], \
            df_surebets['p_hand20_X'], df_surebets['p_hand20_2'], df_surebets['p_hand10_1'], df_surebets['p_hand10_X'], \
            df_surebets['p_hand10_2']
            df_surebets = df_surebets.sort_values(by=['profit_in_%', 'match'], ascending=False)
            df_surebets = df_surebets.drop_duplicates(subset=['bookie', 'match', 'eur'], keep='last')
        return df_surebets
