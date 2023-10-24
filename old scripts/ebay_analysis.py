import warnings
import pandas as pd
import spacy
import scipy.stats as stats
from scipy.stats import ttest_ind
from sklearn import linear_model
from random import choice
from requests.exceptions import ChunkedEncodingError
from geojson import Feature, FeatureCollection, Point
import shapely.wkt
import branca
from simpledbf import Dbf5
import webbrowser
from shapely.geometry import Polygon
import folium
from IPython.display import display
from folium import plugins
from folium.plugins import HeatMap
import plotly.figure_factory as ff
import plotly.tools as tls
import plotly.graph_objs as go
import plotly
import plotly.express as px
import _plotly_geo
from plotly.figure_factory._county_choropleth import create_choropleth
from plotly.offline import init_notebook_mode, iplot
from folium import Choropleth, Circle, Marker
import json
from pandas.io.json import json_normalize
import requests
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_line, geom_boxplot
import shapefile
import requests
from io import BytesIO
from typing import Dict, Any
import spacy.lang.de
import spacy.cli
import textstat
import random
from spacy.util import minibatch
from spacy.training.example import Example
import seaborn as sns
import re
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import cv2
from tqdm import tqdm
plt.rcParams["font.family"] = "Times New Roman"
#spacy.cli.download("de_core_news_sm")
#init_notebook_mode(connected=True)
#plotly.offline.init_notebook_mode(connected=True)


def main():

    warnings.filterwarnings("ignore")
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    apple_offers = pd.read_csv('csv_data/profiles_offers_apple_clean.csv')
    checked_results = pd.read_csv('csv_data/checked_results.csv')
    apple_offers = apple_offers.head(len(checked_results))
    checked_results = checked_results.loc[checked_results['fraud'] != 2].reset_index()
    print('Frauds til now: ' + str(len(checked_results.loc[checked_results['fraud'] == 1].reset_index())) + ' by a verified count of: ' + str(len(checked_results)))
    apple_offers = pd.merge(apple_offers, checked_results, on='offer-ID', how='left')
    del apple_offers['views'], apple_offers['name'], apple_offers['apple_url'], apple_offers['apple-offer-url'], apple_offers['scrape_time']

    ''' APPLE OFFERS ONLY - EXTENDING DATASET '''
    for i in range(0, 1):
        apple_offers = latlong(apple_offers)                                    # Location information (Berlin Kreuzberg, Berlin,  52.5323,  13.3846)
        mainproductclass = product_classifiing1(apple_offers)                   # Product class (iphone)
        detaillproductclass = product_classifiing2(mainproductclass)            # Exact product model (iphone 12 pro max)
        mostdetaillproductclass = product_classifiing3(detaillproductclass)     # Capacity (128)

        #titles_analysed = titles_analysis(apple_offers)                         # Written expressions (emojis, uppers,..)
        #descriptions_analysed = descriptions_analysis(apple_offers)             # Spelling & written expressions (verkauven, emojis,..)
        #descriptions_predicted = descriptions_analysis2(apple_offers)           # Natural Language Processing for honest and fraud
        #pictures_analysed = pictures_analysis(apple_offers)                     # Count of images & reverse image search with google (zenserp api)
        location_analysed = location_analysis(apple_offers)                     # Relationship with lat/lng/plz/bundesland and fraud
        #times_analysed = times_analysis(apple_offers)
        #profile_analysed = profiles_analysis(apple_offers)


        ''' EXPLORATIVE ANALYSIS '''
        for i in range(0, 1):
            #product_classes = product_classes_plot(mostdetaillproductclass)                         # plot iphone models, price & frauds
            #titles = titles_plot(titles_analysed)
            #descriptions1 = descriptions1_plot()
            #descriptions2 = descriptions2_plot(descriptions_predicted)
            #pictures = pictures_plot(apple_offers)
            locations = location_plot(location_analysed)
            #offer_times = time_scraping_plot(apple_offers)                               # plot offer dates & frauds
            #profiles = profile_plot(profile_analysed)


            ''' FINAL DATASET '''
            for i in range(0, 1):
                #del mostdetaillproductclass['fraud'], mostdetaillproductclass['capacity'], mostdetaillproductclass['ipad air'], mostdetaillproductclass['ipad mini']
                #del mostdetaillproductclass['macbook air'], mostdetaillproductclass['watch'], mostdetaillproductclass['iphone 14 (no pro max)']
                # PRODUCT CLASSES & TITLES
                final_data = pd.merge(mostdetaillproductclass, titles_analysed, on='offer-ID', how='left')
                #del final_data['product_model'], final_data['product'], final_data['scraptime'], final_data['fraud'], final_data['emojis_share'], final_data['emojis']
                #del final_data['text'], final_data['numb_count'], final_data['exclam_mark_count'], final_data['exclam_mark_share'], final_data['upper_count']
                #final_data.rename(columns={'length': 'title_length', 'numb_share': 'title_numb_share', 'upper_share': 'title_upper_share'}, inplace=True)
                # PICTURES
                #del final_data['pics'], final_data['titles'], final_data['dates'], final_data['descriptions'], final_data['STADT'], final_data['BUNDESLAND']
                # TIMING
                #del final_data['startweekday'], final_data['startmonth'], final_data['startdayofmonth'], final_data['startyear']
                final_data = pd.merge(final_data, times_analysed, on='offer-ID', how='left')
                #del final_data['offer_day'], final_data['startweekday'], final_data['startmonth'], final_data['startdayofmonth'], final_data['fraud']
                # PROFILES
                #del final_data['profilefollowers'], final_data['profilereliability'], final_data['shippings'], final_data['offeringsonline'], final_data['prices']
                #del final_data['user-ID'], final_data['Sicher_bezahlen'], final_data['offeringssum'], final_data['profilerating'], final_data['Gewerblicher_user']
                #del final_data['profilefriendliness'], final_data['profilereplyrate'], final_data['profilereplyspeed'], final_data['index'], final_data['PLZ'], final_data['LAT'], final_data['LNG']
                final_data = pd.merge(final_data, profile_analysed, on='offer-ID', how='left')
                #del final_data['profilefollowers'], final_data['profilereliability'], final_data['shippings'], final_data['offeringsonline'], final_data['prices']
                # LOCATIONS
                final_data = pd.merge(final_data, location_analysed, on='offer-ID', how='left')
                #del final_data['NUTS3'], final_data['NUTS2'], final_data['einwohner'], final_data['qkm'], final_data['prices'], final_data['west'], final_data['fraud']
                #del final_data['PLZ'], final_data['STADT'], final_data['BUNDESLAND'], final_data['LAT'], final_data['LNG'], final_data['pop_dichte']
                # DESCRIPTIONS
                descriptions = pd.read_csv('csv_data/descriptions_copy.csv')
                final_data = pd.merge(final_data, descriptions, on='offer-ID', how='left')
                #del final_data['characters'], final_data['numbs'], final_data['exclam_marks'], final_data['uppers'], final_data['emojis']
                #del final_data['emojis_share'], final_data['spacy_spelling'], final_data['words'], final_data['text']

                with open('final_dataset.textmate', 'w') as file:
                    file.write(str(final_data) + '\n')


                ''' STATS FINAL DATASET '''
                for i in range(0, 1):
                    #del final_data['offer-ID']
                    # PLOTS
                    corr = final_data.corr(method='pearson')
                    plt.figure(figsize=(20, 6))
                    sns.heatmap(corr, annot=True, cmap='coolwarm_r')
                    plt.title('Correlation matrix')
                    plt.xticks(rotation=40)
                    plt.show()



''' EXPLORATIVE ANALYSIS '''


def latlong(apple_offers):

    latlong = pd.read_table('csv_data/DE.txt', header=None)
    latlong.rename(columns={0: 'LAND', 1: 'PLZ', 2: 'STADT', 3: 'BUNDESLAND', 7: 'KREIS', 8: 'VORWAHL', 9: 'LAT', 10: 'LNG'}, inplace=True)
    latlong['PLZ'] = latlong['PLZ'].astype(str)
    latlong['PLZ'] = latlong['PLZ'].drop_duplicates(keep='first')
    latlong['PLZ'] = latlong['PLZ'].astype(str)
    latlong = latlong.loc[latlong['PLZ'] != 'nan']
    latlong = latlong.reset_index()
    del latlong['index'], latlong['LAND'], latlong[11], latlong[4], latlong[5], latlong[6], latlong['VORWAHL'], latlong['KREIS']
    plz = []
    for i in latlong['PLZ']:
        plz.append(i)
    latlong['PLZ'] = plz
    latlong['PLZ'] = latlong['PLZ'].astype(int)
    apple_offers['PLZ'] = apple_offers['locations'].astype(int)
    del apple_offers['locations']
    df1 = pd.merge(apple_offers, latlong, on='PLZ', how='left')

    plz = []
    for i in df1['PLZ']:
        plz.append(i)
    df1['PLZ'] = plz
    df1['PLZ'] = df1['PLZ']
    df1 = df1.dropna(subset='fraud')

    df1 = df1.sort_values('PLZ', ascending=True)
    df1.fillna(method='ffill', inplace=True)
    df1.fillna(method='bfill', inplace=True)
    df1 = df1.loc[(df1['prices'] > 50) & (df1['prices'] < 2800) & (df1['PLZ'] < 99999)]
    df1 = df1.reset_index()
    del df1['level_0']
    with open('latlong.textmate', 'w') as file:
        file.write(str(df1) + '\n')

    return df1


def product_classifiing1(apple_offers):

    main_product_class = []
    for i in apple_offers['titles']:
        i = i.lower()
        if 'phon' in i:
            main_product_class.append('iphone')
        elif 'pod' in i:
            main_product_class.append('airpods')
        elif 'pad' in i:
            main_product_class.append('ipad')
        elif 'book' in i:
            main_product_class.append('macbook')
        elif 'mac' in i:
            main_product_class.append('mac')
        elif 'watch' in i:
            main_product_class.append('watch')
        else:
            main_product_class.append('other')
    apple_offers['product'] = main_product_class

    return apple_offers


def product_classifiing2(mainproductclass):

    detailled_product_class = []
    for i in mainproductclass['titles']:
        i = i.lower()

        # IPHONES
        if 'phon' in i and '14 ' in i and 'max' in i and 'pro' in i:
            detailled_product_class.append('iphone 14 pro max')
        elif 'phon' in i and '14 ' in i and 'pro' in i:
            detailled_product_class.append('iphone 14 pro')
        elif 'phon' in i and '14 ' in i and 'plus' in i:
            detailled_product_class.append('iphone 14 plus')
        elif 'phon' in i and '14 ' in i:
            detailled_product_class.append('iphone 14')

        elif 'phon' in i and '13 ' in i and 'max' in i and 'pro' in i:
            detailled_product_class.append('iphone 13 pro max')
        elif 'phon' in i and '13 ' in i and 'pro' in i:
            detailled_product_class.append('iphone 13 pro')
        elif 'phon' in i and '13 ' in i and 'mini' in i:
            detailled_product_class.append('iphone 13 mini')
        elif 'phon' in i and '13 ' in i:
            detailled_product_class.append('iphone 13')

        elif 'phon' in i and 'xr ' in i:
            detailled_product_class.append('iphone xr')
        elif 'phon' in i and 'xs ' in i:
            detailled_product_class.append('iphone xs')
        elif 'phon' in i and ' x ' in i:
            detailled_product_class.append('iphone x')

        elif 'phon' in i and '11 ' in i and 'max' in i and 'pro' in i:
            detailled_product_class.append('iphone 11 pro max')
        elif 'phon' in i and '11 ' in i and 'pro' in i:
            detailled_product_class.append('iphone 11 pro')
        elif 'phon' in i and '11 ' in i:
            detailled_product_class.append('iphone 11')

        elif 'phon' in i and '7 ' in i and 'plus' in i:
            detailled_product_class.append('iphone 7 plus')
        elif 'phon' in i and '7 ' in i:
            detailled_product_class.append('iphone 7')
        elif 'phon' in i and ' 8' in i and 'plus' in i:
            detailled_product_class.append('iphone 8 plus')
        elif 'phon' in i and ' 8' in i:
            detailled_product_class.append('iphone 8')

        elif 'phon' in i and '12 ' in i and 'max' in i and 'pro' in i:
            detailled_product_class.append('iphone 12 pro max')
        elif 'phon' in i and '12 ' in i and 'pro' in i:
            detailled_product_class.append('iphone 12 pro')
        elif 'phon' in i and '12 ' in i and 'mini' in i:
            detailled_product_class.append('iphone 12 mini')
        elif 'phon' in i and '12 ' in i:
            detailled_product_class.append('iphone 12')

        elif 'phon' in i:
            detailled_product_class.append('iphone')

        # AIRPODS
        elif 'pod' in i and 'max' in i:
            detailled_product_class.append('airpods max')
        elif 'pod' in i and 'pro' in i:
            detailled_product_class.append('airpods 2 pro')
        elif 'pod' in i and '2' in i:
            detailled_product_class.append('airpods 2')
        elif 'pod' in i and '3' in i:
            detailled_product_class.append('airpods 3')
        elif 'pod' in i:
            detailled_product_class.append('airpods')

        # IPADS
        elif 'pad' in i and 'pro' in i:
            detailled_product_class.append('ipad pro')
        elif 'pad' in i and 'mini' in i:
            detailled_product_class.append('ipad mini')
        elif 'pad' in i and 'air' in i:
            detailled_product_class.append('ipad air')
        elif 'pad' in i:
            detailled_product_class.append('ipad')

        # APPLE WATCHES
        elif 'watch' in i and 'ultra' in i:
            detailled_product_class.append('watch ultra')
        elif 'watch' in i and 'series' in i:
            detailled_product_class.append('watch series')
        elif 'watch' in i and 'se' in i:
            detailled_product_class.append('watch se')
        elif 'watch' in i:
            detailled_product_class.append('watch')

        # MACBOOKS
        elif 'book' in i and 'air' in i and '2022' in i:
            detailled_product_class.append('macbook air 2022')
        elif 'book' in i and 'air' in i and '2021' in i:
            detailled_product_class.append('macbook air 2021')
        elif 'book' in i and 'air' in i and '2020' in i:
            detailled_product_class.append('macbook air 2020')
        elif 'book' in i and 'air' in i and '2019' in i:
            detailled_product_class.append('macbook air 2019')
        elif 'book' in i and 'air' in i:
            detailled_product_class.append('macbook air')
        elif 'book' in i and 'pro' in i and '2022' in i:
            detailled_product_class.append('macbook pro 2022')
        elif 'book' in i and 'pro' in i and '2021' in i:
            detailled_product_class.append('macbook pro 2021')
        elif 'book' in i and 'pro' in i and '2020' in i:
            detailled_product_class.append('macbook pro 2020')
        elif 'book' in i and 'pro' in i and '2019' in i:
            detailled_product_class.append('macbook pro 2019')
        elif 'book' in i and 'pro' in i:
            detailled_product_class.append('macbook pro')
        elif 'book' in i:
            detailled_product_class.append('macbook')

        # PEN
        elif 'pen' in i:
            detailled_product_class.append('pencil')

        # KEYBOARD
        elif 'key' in i:
            detailled_product_class.append('keyboard')

        # IMAC
        elif 'imac' in i:
            detailled_product_class.append('imac')
        elif 'i mac' in i:
            detailled_product_class.append('imac')

        # MAC
        elif 'mac' in i and 'mini' in i:
            detailled_product_class.append('mac mini')
        elif 'mac' in i:
            detailled_product_class.append('imac')

        # TV
        elif 'tv' in i:
            detailled_product_class.append('tv')

        # OTHER
        else:
            detailled_product_class.append('other')
    mainproductclass['product_model'] = detailled_product_class
    # CREATE DUMMIES FOR PRODUCTCLASSES, CLASSIFIER JUST WORK WITH NUMERIC DATA
    X_cat = pd.get_dummies(mainproductclass['product_model'], drop_first=True)
    X_cat['others'] = X_cat['other'] + X_cat['pencil'] + X_cat['tv'] + X_cat['imac'] + X_cat['keyboard'] + X_cat['mac mini'] + X_cat['macbook'] + X_cat['ipad']
    X_cat['macbook air'] = X_cat['macbook air'] + X_cat['macbook air 2020'] + X_cat['macbook air 2021'] + X_cat['macbook air 2022']
    X_cat['macbook pro'] = X_cat['macbook pro'] + X_cat['macbook pro 2019'] + X_cat['macbook pro 2020'] + X_cat['macbook pro 2021'] + X_cat['macbook pro 2022']
    X_cat['watch'] = X_cat['watch'] + X_cat['watch se'] + X_cat['watch series'] + X_cat['watch ultra']
    X_cat['old iphone'] = X_cat['iphone x'] + X_cat['iphone xr'] + X_cat['iphone xs'] + X_cat['iphone 7'] + X_cat['iphone 7 plus'] + X_cat['iphone 8'] + X_cat['iphone 8 plus'] + X_cat['iphone']
    X_cat['old airpods'] = X_cat['airpods 2'] + X_cat['airpods 3']
    X_cat['new airpods'] = X_cat['airpods 2 pro'] + X_cat['airpods max']
    X_cat['iphone 14 (no pro max)'] = X_cat['iphone 14'] + X_cat['iphone 14 pro'] + X_cat['iphone 14 plus']
    X_cat['iphone 13 (no pro max)'] = X_cat['iphone 13'] + X_cat['iphone 13 pro'] + X_cat['iphone 13 mini']
    X_cat['iphone 12'] = X_cat['iphone 12'] + X_cat['iphone 12 mini'] + X_cat['iphone 12 pro'] + X_cat['iphone 12 pro max']
    X_cat['iphone 11'] = X_cat['iphone 11'] + X_cat['iphone 11 pro'] + X_cat['iphone 11 pro max']
    del X_cat['pencil'], X_cat['other'], X_cat['tv'], X_cat['imac'], X_cat['keyboard'], X_cat['mac mini'], X_cat['macbook'], X_cat['ipad']
    del X_cat['macbook air 2020'], X_cat['macbook air 2021'], X_cat['macbook air 2022']
    del X_cat['macbook pro 2019'], X_cat['macbook pro 2020'], X_cat['macbook pro 2021'], X_cat['macbook pro 2022']
    del X_cat['watch se'], X_cat['watch series'], X_cat['watch ultra']
    del X_cat['iphone x'], X_cat['iphone xr'], X_cat['iphone xs'], X_cat['iphone 7'], X_cat['iphone 7 plus'], X_cat['iphone 8'], X_cat['iphone 8 plus'], X_cat['iphone']
    del X_cat['airpods 2'], X_cat['airpods 3']
    del X_cat['airpods 2 pro'], X_cat['airpods max']
    del X_cat['iphone 14'], X_cat['iphone 14 pro'], X_cat['iphone 14 plus']
    del X_cat['iphone 13'], X_cat['iphone 13 pro'], X_cat['iphone 13 mini']
    del X_cat['iphone 12 mini'], X_cat['iphone 12 pro'], X_cat['iphone 12 pro max']
    del X_cat['iphone 11 pro'], X_cat['iphone 11 pro max']
    X_cat['offer-ID'] = mainproductclass['offer-ID']
    mainproductclass = pd.merge(mainproductclass, X_cat, on='offer-ID', how='left')
    mainproductclass['ipad air'] = mainproductclass['ipad air'].astype(int)
    mainproductclass['ipad mini'] = mainproductclass['ipad mini'].astype(int)
    mainproductclass['ipad pro'] = mainproductclass['ipad pro'].astype(int)
    mainproductclass['iphone 11'] = mainproductclass['iphone 11'].astype(int)
    mainproductclass['iphone 12'] = mainproductclass['iphone 12'].astype(int)
    mainproductclass['iphone 13 (no pro max)'] = mainproductclass['iphone 13 (no pro max)'].astype(int)
    mainproductclass['iphone 13 pro max'] = mainproductclass['iphone 13 pro max'].astype(int)
    mainproductclass['iphone 14 (no pro max)'] = mainproductclass['iphone 14 (no pro max)'].astype(int)
    mainproductclass['iphone 14 pro max'] = mainproductclass['iphone 14 pro max'].astype(int)
    mainproductclass['macbook air'] = mainproductclass['macbook air'].astype(int)
    mainproductclass['macbook pro'] = mainproductclass['macbook pro'].astype(int)
    mainproductclass['watch'] = mainproductclass['watch'].astype(int)
    mainproductclass['others'] = mainproductclass['others'].astype(int)
    mainproductclass['old iphone'] = mainproductclass['old iphone'].astype(int)
    mainproductclass['old airpods'] = mainproductclass['old airpods'].astype(int)
    mainproductclass['new airpods'] = mainproductclass['new airpods'].astype(int)

    return mainproductclass


def product_classifiing3(detaillproductclass):

    most_detailled_product_class = []
    for i in detaillproductclass['titles']:
        i = i.lower()
        # IPHONES
        if 'phon' in i and '16' in i:
            most_detailled_product_class.append(16)
        elif 'phon' in i and '32' in i:
            most_detailled_product_class.append(32)
        elif 'phon' in i and '64' in i:
            most_detailled_product_class.append(64)
        elif 'phon' in i and '128' in i:
            most_detailled_product_class.append(128)
        elif 'phon' in i and '256' in i:
            most_detailled_product_class.append(256)
        elif 'phon' in i and '512' in i:
            most_detailled_product_class.append(512)
        elif 'phon' in i:
            most_detailled_product_class.append(None)

        # IPADS
        elif 'pad' in i and '16' in i:
            most_detailled_product_class.append(16)
        elif 'pad' in i and '32' in i:
            most_detailled_product_class.append(32)
        elif 'pad' in i and '64' in i:
            most_detailled_product_class.append(64)
        elif 'pad' in i and '128' in i:
            most_detailled_product_class.append(128)
        elif 'pad' in i and '256' in i:
            most_detailled_product_class.append(256)
        elif 'pad' in i and '512' in i:
            most_detailled_product_class.append(512)
        elif 'pad' in i:
            most_detailled_product_class.append(None)

        # MACBOOKS
        elif 'book' in i and '128' in i:
            most_detailled_product_class.append(128)
        elif 'book' in i and '256' in i:
            most_detailled_product_class.append(256)
        elif 'book' in i and '512' in i:
            most_detailled_product_class.append(512)
        elif 'book' in i and '1TB' in i:
            most_detailled_product_class.append(1000)
        elif 'book' in i and '1 TB' in i:
            most_detailled_product_class.append(1000)
        elif 'book' in i and '2TB' in i:
            most_detailled_product_class.append(2000)
        elif 'book' in i and '2 TB' in i:
            most_detailled_product_class.append(2000)
        elif 'book' in i:
            most_detailled_product_class.append(None)

        # MACBOOKS
        elif 'mac' in i and '128' in i:
            most_detailled_product_class.append(128)
        elif 'mac' in i and '256' in i:
            most_detailled_product_class.append(256)
        elif 'mac' in i and '512' in i:
            most_detailled_product_class.append(512)
        elif 'mac' in i and '1TB' in i:
            most_detailled_product_class.append(1000)
        elif 'mac' in i and '1 TB' in i:
            most_detailled_product_class.append(1000)
        elif 'mac' in i and '2TB' in i:
            most_detailled_product_class.append(2000)
        elif 'mac' in i and '2 TB' in i:
            most_detailled_product_class.append(2000)
        elif 'mac' in i:
            most_detailled_product_class.append(None)

        else:
            most_detailled_product_class.append(None)
    detaillproductclass['capacity'] = most_detailled_product_class
    detaillproductclass['capacity'] = detaillproductclass['capacity'].fillna(detaillproductclass['capacity'].mean())
    detaillproductclass['capacity_log'] = np.log(detaillproductclass['capacity'])
    with open('mostdetaillproductclass.textmate', 'w') as file:
        file.write(str(detaillproductclass) + '\n')

    return detaillproductclass


def titles_analysis(apple_offers):

    nlp = spacy.load("de_core_news_sm")
    nlp.add_pipe("emoji", first=True)
    df1 = pd.DataFrame()
    for m in range(0, len(apple_offers['titles'])):
        i = apple_offers['titles'][m]
        text = []
        partofspeech = []
        finegrainedpartofspeech = []
        isupper = []
        emoji = []
        doc = nlp(i)
        for token in doc:
            text.append(token)
            partofspeech.append(token.pos_)
            finegrainedpartofspeech.append(spacy.explain(token.tag_))
            isupper.append(token.is_upper)
            emoji.append(token._.is_emoji)


        x = 0
        for j in partofspeech:
            if j == 'NUM':
                x = x+1
        numb_count = x
        x = 0
        for j in finegrainedpartofspeech:
            if j == 'sentence-final punctuation mark':
                x = x + 1
        exclam_mark_count = x
        x = 0
        for j in isupper:
            if j == True:
                x = x + 1
        upper_count = x
        x = 0
        for j in emoji:
            if j == True:
                x = x + 1
        emojis = x

        df = pd.DataFrame()
        df['length'] = [len(i)-21]
        df['text'] = [text]
        df['numb_count'] = [numb_count]
        df['numb_share'] = df['numb_count']/df['length']
        df['exclam_mark_count'] = [exclam_mark_count]
        df['exclam_mark_share'] = df['exclam_mark_count']/df['length']
        df['upper_count'] = [upper_count]
        df['upper_share'] = df['upper_count']/df['length']
        df['emojis'] = [emojis]
        df['emojis_share'] = df['emojis']/df['length']
        df1 = df1.append(df, ignore_index=True)
    df1['offer-ID'] = apple_offers['offer-ID']
    df1['fraud'] = apple_offers['fraud']

    with open('titles.textmate', 'w') as file:
        file.write(str(df1) + '\n')

    return df1


def descriptions_analysis(apple_offers):
    #apple_offers = apple_offers.head(10)

    nlp = spacy.load("de_core_news_sm")
    nlp.add_pipe("emoji", first=True)
    df1 = pd.DataFrame()
    for m in range(0, len(apple_offers['descriptions'])):
        i = apple_offers['descriptions'][m]
        text = []
        partofspeech = []
        finegrainedpartofspeech = []
        isupper = []
        spelling = []
        emoji = []
        doc = nlp(i)
        paypal = 0
        paypal_freunde = 0
        for token in doc:
            if str(token) == 'sofortüberweisung' or str(token) == 'Sofortüberweisung' or str(token) == 'Freunde' or str(token) == 'freunde' or str(token) == 'Friends':
                paypal_freunde = 1
            if str(token) == 'paypal' or str(token) == 'Paypal' or str(token) == 'PayPal':
                paypal = 1
            text.append(token)
            partofspeech.append(token.pos_)
            finegrainedpartofspeech.append(spacy.explain(token.tag_))
            isupper.append(token.is_upper)
            spelling.append(token.vector_norm)
            emoji.append(token._.is_emoji)

        x = 0
        for j in partofspeech:
            if j == 'NUM':
                x = x + 1
        numb_count = x
        x = 0
        for j in finegrainedpartofspeech:
            if j == 'sentence-final punctuation mark':
                x = x + 1
        exclam_mark_count = x
        x = 0
        for j in isupper:
            if j == True:
                x = x + 1
        upper_count = x
        x = 0
        for j in emoji:
            if j == True:
                x = x + 1
        emojis = x
        spelling = sum(spelling)/len(spelling)


        df = pd.DataFrame()
        df['characters'] = [len(i)-33]
        df['text'] = [text]
        df['numbs'] = [numb_count]
        df['numb_share'] = df['numbs']/df['characters']
        df['exclam_marks'] = [exclam_mark_count]
        df['exclam_mark_share'] = df['exclam_marks']/df['characters']
        df['uppers'] = [upper_count]
        df['upper_share'] = df['uppers']/df['characters']
        df['emojis'] = [emojis]
        df['emojis_share'] = df['emojis']/df['characters']
        df['spacy_spelling'] = [spelling]
        df['paypal'] = paypal
        df['paypal_freunde'] = paypal_freunde
        df1 = df1.append(df, ignore_index=True)
        print(m)

    # recognizing grammar mistakes in description
    df_grammar = pd.read_csv('csv_data/de_DE 2.csv', encoding='latin-1')
    df_grammar['Äbte'] = df_grammar['Äbte'].str.lower()
    list = []
    for d in df_grammar['Äbte']:
        list.append(d)
    spelling_list = []
    length = []
    X = df1['text']
    for x in X:
        x = x[2:-2]
        x = str(x)
        x = re.sub(r'[!@#$()"%^*?:/.;~`0-9 ]', '', x)  # removing the symbols and numbers
        x = re.sub(r"['-]", '', x)  # removing the symbols and numbers
        x = re.sub(r'[[]]', ' ', x)
        x = x.lower()
        x = x.split(',')
        text_gesamt = []
        df_writing = []

        for y in x:
            if y != '' or y != '[' or y != ']':
                doc = nlp(y)
                for token in doc:
                    y = token.lemma_
                    y = y.lower()
                text_gesamt.append(y)
                if y in list:
                    df_writing.append(1)
                elif y == '':
                    df_writing.append(1)
                else:
                    df_writing.append(0)
        spelling_list.append(sum(df_writing))
        length.append(len(df_writing))

    df1['correct_words'] = spelling_list
    df1['words'] = length
    df1['mistake_rate'] = 1 - df1['correct_words']/df1['words']
    df1['offer-ID'] = apple_offers['offer-ID']
    del df1['correct_words']
    df1['fraud'] = apple_offers['fraud']
    with open('descriptions.textmate', 'w') as file:
        file.write(str(df1) + '\n')
    df1.to_csv('/Users/philippschenk/PycharmProjects/ebay_scraper2/csv_data/descriptions.csv', index=False)

    return df1


def descriptions_analysis2(apple_offers):

    df1 = pd.DataFrame()
    df1['fraud'] = apple_offers['fraud']
    df1['descriptions'] = apple_offers['descriptions']
    #print(df1.head())
    nlp = spacy.blank("de")
    textcat = nlp.add_pipe("textcat")
    textcat.add_label("honest")
    textcat.add_label("fraud")

    train_texts = df1['descriptions'][:6000].values
    train_labels = df1['fraud'][:6000].values
    train_data = list(zip(train_texts, train_labels))
    print(train_data[:3])

    examples = []
    for x, c in train_data:
        doc = nlp.make_doc(x)
        example = Example.from_dict(doc, {"cats": {"fraud": c}})
        examples.append(example)
    print(examples)

    with nlp.select_pipes(enable="textcat"):
        optimizer = nlp.initialize(lambda: examples)
        print(optimizer)
        for itn in range(5):
            random.shuffle(examples)
            nlp.update(examples, sgd=optimizer)

    texts = df1['descriptions'][6000:]

    docs = [nlp.tokenizer(text) for text in texts]

    # Use textcat to get the scores for each doc
    textcat = nlp.get_pipe('textcat')
    scores = textcat.predict(docs)
    description_fraud_level = []
    for x in scores:
        description_fraud_level.append(x[1])
    predicted_labels = scores.argmax(axis=1)
    df_results = pd.DataFrame()
    df_results['pred_fraud_desc'] = description_fraud_level
    texts = texts.reset_index()
    del texts['index']
    df_results['texts'] = texts
    new_list = df1['fraud'][6000:]
    new_list = new_list.reset_index()
    del new_list['index']
    df_results['fraud'] = new_list
    df_results['fraud2'] = [textcat.labels[label] for label in predicted_labels]

    with open('descriptions2.textmate', 'w') as file:
        file.write(str(df_results) + '\n')

    return df_results


def pictures_analysis(apple_offers):

    old_data = pd.read_csv('csv_data/reverse_image_search.csv')
    del old_data['fraud'], old_data['offer-ID']
    old_data['pics'] = old_data['pics'].astype(str)
    new_list = []
    for i in old_data['pics']:
        new_list.append(i[:75])
    old_data['pic_short'] = new_list

    apple_offers1 = apple_offers.tail(len(apple_offers) - len(old_data))
    apple_offers2 = pd.DataFrame()
    apple_offers2['pics'] = apple_offers['pics']
    apple_offers2['pics'] = apple_offers2['pics'].astype(str)
    apple_offers2['pics'] = apple_offers['pics']
    apple_offers2['fraud'] = apple_offers['fraud']
    apple_offers2['offer-ID'] = apple_offers['offer-ID']
    new_list = []
    for i in apple_offers2['pics']:
        new_list.append(i[:75])
    apple_offers2['pic_short'] = new_list

    pictures_data = pd.merge(old_data, apple_offers2, on='pic_short', how='left')
    del pictures_data['pics_y'], pictures_data['pic_short']
    with open('pictures_data.textmate', 'w') as file:
        file.write(str(pictures_data) + '\n')


    for l in apple_offers1['pics']:
        print(l)
        similar_images = []
        pages_with_matching_images = []
        datasets = []
        pic_count = []
        pages_with_matching_images_urls = []
        dataset = l
        old_data = pd.read_csv('csv_data/reverse_image_search.csv')
        if dataset not in old_data['pics']:
            dataset = dataset.replace('[', '')
            dataset = dataset.replace(']', '')
            dataset = dataset.replace("'", '')
            dataset = dataset.replace('"', '')
            dataset = dataset.split(',')
            length1 = []
            length2 = []
            pages_with_matching_images_url = []
            if len(dataset) > 0 and dataset != ['']:
                for i in dataset:
                    print(i)
                    older_data = pd.read_csv('csv_data/reverse_image_search.csv')
                    url = i
                    params = (("image_url", str(url)),("gl", "DE"),("hl", "de"),)

                    response = requests.get('https://app.zenserp.com/api/v2/search', headers=headers, params=params)
                    if 'A timeout occurred' in response.text:
                        length1.append(1)
                        length2.append(1)
                        pages_with_matching_images_url.append(1)
                        break
                    else:
                        response_dict = json.loads(response.text)
                        print(len(response_dict['reverse_image_results']['similar_images']))
                        print(len(response_dict['reverse_image_results']['pages_with_matching_images']))
                        if 'reverse_image_results' in response.text:
                            length1.append(len(response_dict['reverse_image_results']['similar_images']))
                            length2.append(len(response_dict['reverse_image_results']['pages_with_matching_images']))
                            pages_with_matching_images_url.append(response_dict['reverse_image_results']['pages_with_matching_images'])

                similar_images.append(sum(length1)/len(length1))
                pages_with_matching_images.append(sum(length2)/len(length2))
                datasets.append(dataset)
                pic_count.append(len(dataset))
                pages_with_matching_images_urls.append(pages_with_matching_images_url)

            df222 = pd.DataFrame()
            df222['pics'] = datasets
            df222['pic_count'] = pic_count
            df222['similar_images'] = similar_images
            df222['pages_with_matching_images'] = pages_with_matching_images
            df222['pages_with_matching_images_urls'] = pages_with_matching_images_urls

            older_data = older_data.append(df222, ignore_index=True)
            older_data['offer-ID'] = apple_offers['offer-ID'].head(len(older_data))
            older_data['fraud'] = apple_offers['fraud'].head(len(older_data))
            older_data = older_data.drop_duplicates(subset=['offer-ID'], keep='first')
            print(older_data)
            older_data.to_csv('csv_data/reverse_image_search.csv', index=False)

    return older_data


def location_analysis(apple_offers):

    nuts2_de = pd.read_csv('csv_data/pc2020_DE_NUTS-2021_v1.0.csv', delimiter=';')
    pop_density = pd.read_csv('csv_data/bahn.csv')
    crime_rate_nuts3 = pd.read_csv('csv_data/crime_rate_nuts3.csv', delimiter=';')
    dbf2 = Dbf5('csv_data/plz-5stellig.dbf', codec='latin')
    df2 = dbf2.to_dataframe()

    df1 = pd.DataFrame()
    df1['fraud'] = apple_offers['fraud']
    df1['offer-ID'] = apple_offers['offer-ID']
    df1['PLZ'] = apple_offers['PLZ'].astype(int)
    df1['STADT'] = apple_offers['STADT']
    df1['BUNDESLAND'] = apple_offers['BUNDESLAND']
    df1['LAT'] = apple_offers['LAT']
    df1['LNG'] = apple_offers['LNG']
    df1['prices'] = apple_offers['prices']

    nuts2 = []
    nuts3 = []
    plz = []
    for x in nuts2_de['NUTS3']:
        nuts3.append(x[1:6])
        nuts2.append(x[1:5])
    for x in nuts2_de['PLZ']:
        plz.append(x[1:-1])

    nuts2_de['NUTS3'] = nuts3
    nuts2_de['NUTS2'] = nuts2
    nuts2_de['PLZ'] = plz
    nuts2_de['PLZ'] = nuts2_de['PLZ'].astype(int)
    df = pd.merge(df1, nuts2_de, on='PLZ', how='left')


    pop_density = pop_density.loc[pop_density['TIME_PERIOD'] == 2018]
    pop_density = pop_density.reset_index()
    del pop_density['DATAFLOW'], pop_density['LAST UPDATE'], pop_density['freq'], pop_density['unit'], pop_density['TIME_PERIOD'], pop_density['OBS_FLAG'], pop_density['index']
    df1 = pd.merge(df, pop_density, on='NUTS2', how='left')
    df1['pop_dichte'] = df1['POP_DICHTE'].astype(float)
    del df1['POP_DICHTE']

    del crime_rate_nuts3['GEO (Labels)'], crime_rate_nuts3['2008'], crime_rate_nuts3['2009']
    crime_rate_nuts3 = crime_rate_nuts3.rename(columns={'2010': 'crime_rate'})
    df = pd.merge(df1, crime_rate_nuts3, on='NUTS3', how='left')
    df['crime_rate'] = df['crime_rate'].astype(float)

    del df2['note']
    df2 = df2.rename(columns={'plz': 'PLZ'})
    df2['PLZ'] = df2['PLZ'].astype(int)
    df = pd.merge(df, df2, on='PLZ', how='left')
    df['einwohner'] = df['einwohner'].astype(float)
    df['qkm'] = df['qkm'].astype(float)
    df['PLZ'] = df['PLZ'].astype(int)
    df['LAT'] = df['LAT'].astype(float)
    df['LNG'] = df['LNG'].astype(float)
    df = df.loc[df['PLZ'] < 99999]
    #df['fraud_per_capita'] = df['fraud'] / df['einwohner']

    df = df.sort_values(['PLZ'], ascending=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    new_bl = []
    for x in df['BUNDESLAND']:
        x = x.replace('Baden-Württemberg', 'Baden_Wuerttemberg')
        new_bl.append(x)
    df['BUNDESLAND'] = new_bl

    X_cat = pd.get_dummies(df['BUNDESLAND'])
    X_cat['east'] = X_cat['Berlin'] + X_cat['Brandenburg'] + X_cat['Sachsen'] + X_cat['Sachsen-Anhalt'] + X_cat['Thüringen'] + X_cat['Mecklenburg-Vorpommern']
    X_cat['west'] = X_cat['Hamburg'] + X_cat['Niedersachsen'] + X_cat['Schleswig-Holstein'] + X_cat['Bremen'] + X_cat['Nordrhein-Westfalen'] + X_cat['Hessen'] + X_cat['Rheinland-Pfalz'] + X_cat['Saarland'] + X_cat['Baden_Wuerttemberg'] + X_cat['Bayern']

    del X_cat['Berlin'], X_cat['Brandenburg'], X_cat['Sachsen'], X_cat['Sachsen-Anhalt'], X_cat['Thüringen'], X_cat['Mecklenburg-Vorpommern']
    del X_cat['Hamburg'], X_cat['Niedersachsen'], X_cat['Schleswig-Holstein'], X_cat['Bremen'], X_cat['Nordrhein-Westfalen'], X_cat['Hessen'], X_cat['Rheinland-Pfalz'], X_cat['Saarland'], X_cat['Baden_Wuerttemberg'], X_cat['Bayern']
    X_cat['offer-ID'] = df['offer-ID']
    df = pd.merge(df, X_cat, on='offer-ID', how='left')
    df['east'] = df['east'].astype(int)
    df['west'] = df['west'].astype(int)

    return df


def times_analysis(apple_offers):
    df2 = apple_offers
    day = []
    month = []
    year = []
    for i in df2['dates']:
        i = i.split('.')
        day.append(i[0])
        month.append(i[1])
        year.append(i[2])
    df2['day'] = day
    df2['day'] = df2['day'].astype(int)
    df2['month'] = month
    df2['month'] = df2['month'].astype(int)
    df2['year'] = year
    df2['year'] = df2['year'].astype(int)

    df3 = pd.DataFrame()
    df3['startyear'] = df2['startyear']
    df3['startweekday'] = df2['startweekday']
    df3['startmonth'] = df2['startmonth']
    df3['startdayofmonth'] = df2['startdayofmonth']
    df3['offer_day'] = df2['day']
    df3['offer-ID'] = df2['offer-ID']
    df3['fraud'] = df2['fraud']
    return df3


def profiles_analysis(apple_offers):

    df = pd.DataFrame()
    df['fraud'] = apple_offers['fraud']
    df['prices'] = apple_offers['prices']
    df['shippings'] = apple_offers['shippings']
    df['Sicher_bezahlen'] = apple_offers['Sicher_bezahlen']
    df['Gewerblicher_user'] = apple_offers['Gewerblicher_user']
    df['offeringsonline'] = apple_offers['offeringsonline']
    df['offeringssum'] = apple_offers['offeringssum']
    df['profilerating'] = apple_offers['profilerating']
    df['profilefriendliness'] = apple_offers['profilefriendliness']
    df['profilereliability'] = apple_offers['profilereliability']
    df['profilereplyrate'] = apple_offers['profilereplyrate']
    df['profilereplyspeed'] = apple_offers['profilereplyspeed']
    df['profilefollowers'] = apple_offers['profilefollowers']
    df['offer-ID'] = apple_offers['offer-ID']
    return df


''' PLOTTING '''


def product_classes_plot(mostdetaillproductclass):

    # PRODUCT MODEL COUNT
    mostdetaillproductclass['PLZ'] = mostdetaillproductclass['PLZ'].astype(int)
    mostdetaillproductclass = mostdetaillproductclass.loc[(mostdetaillproductclass['prices'] > 50) & (mostdetaillproductclass['prices'] < 2000) & (mostdetaillproductclass['PLZ'] < 99999)]
    df2 = mostdetaillproductclass#.loc[(mostdetaillproductclass['product'] == 'iphone') & (mostdetaillproductclass['product_model'] != 'iphone')]
    df2 = df2['product_model'].value_counts().to_frame()
    df2 = df2.reset_index()
    df2.columns = ['product_model', 'count']
    fig, ax = plt.subplots(figsize=(3, 1))
    sns.barplot(data=df2, y='product_model', x='count', color='#6693F5')
    plt.title("Product model count")


    # FRAUD SHARE
    df3 = mostdetaillproductclass#.loc[(mostdetaillproductclass['product'] == 'iphone') & (mostdetaillproductclass['product_model'] != 'iphone')]
    fraud_level = []
    prices = []
    for i in df2['product_model']:
        df4 = df3.loc[df3['product_model'] == str(i)]
        fraud_level.append(sum(df4['fraud']))
        prices.append(sum(df4['prices'])/len(df4))
    df2['frauds'] = fraud_level
    df2['fraud_share'] = df2['frauds']/df2['count']
    df2['prices'] = prices
    df2 = df2.sort_values(by='fraud_share', ascending=False)
    fig, ax = plt.subplots(figsize=(3, 1))
    sns.barplot(data=df2, x="fraud_share", y="product_model", color='#6693F5')
    plt.title("Fraud by product model")


    # AIRPODS: FRAUD SHARE
    df222 = mostdetaillproductclass.loc[(mostdetaillproductclass['product'] == 'airpods') & (mostdetaillproductclass['product_model'] != 'airpods')]
    df222 = df222['product_model'].value_counts().to_frame()
    df222 = df222.reset_index()
    df222.columns = ['product_model', 'count']
    df333 = mostdetaillproductclass.loc[(mostdetaillproductclass['product'] == 'airpods') & (mostdetaillproductclass['product_model'] != 'airpods')]
    fraud_level = []
    prices = []
    for i in df222['product_model']:
        df444 = df333.loc[df333['product_model'] == str(i)]
        fraud_level.append(sum(df444['fraud']))
        prices.append(sum(df444['prices']) / len(df444))
    df222['frauds'] = fraud_level
    df222['fraud_share'] = df222['frauds'] / df222['count']
    df222['prices'] = prices
    df222 = df222.sort_values(by='fraud_share', ascending=False)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df222, x="fraud_share", y="product_model", color='#6693F5')
    plt.title("Fraud by product model (airpods)")


    # ALL APPLE PRODUCTS: FRAUD SHARE BY PRODUCT
    df22 = mostdetaillproductclass
    df22 = df22['product'].value_counts().to_frame()
    df22 = df22.reset_index()
    df22.columns = ['product', 'count']
    df33 = mostdetaillproductclass
    fraud_level = []
    prices = []
    for i in df22['product']:
        df44 = df33.loc[df33['product'] == str(i)]
        fraud_level.append(sum(df44['fraud']))
        prices.append(sum(df44['prices']) / len(df44))
    df22['frauds'] = fraud_level
    df22['fraud_share'] = df22['frauds'] / df22['count']
    df22['prices'] = prices
    df22 = df22.sort_values(by='fraud_share', ascending=False)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df22, x="fraud_share", y="product", color='#6693F5')
    plt.title("Fraud by product")


    # PRICE LEVELS
    df2 = df2.sort_values(by='prices', ascending=False)
    fig, ax = plt.subplots(figsize=(3, 1))
    sns.barplot(data=df2, x="prices", y="product_model", color='#6693F5')
    plt.title("Mean prices by product")


    ''''''''' upper: products & productclasses; lower: capacity levels '''''''''

    # CAPACITY
    df2222 = mostdetaillproductclass
    df2222 = df2222['capacity'].value_counts().to_frame()
    df2222 = df2222.reset_index()
    df2222.columns = ['capacity', 'count']
    df3333 = mostdetaillproductclass
    fraud_level = []
    prices = []
    for i in df2222['capacity']:
        df4444 = df3333.loc[df3333['capacity'] == float(i)]
        fraud_level.append(sum(df4444['fraud']))
        prices.append(sum(df4444['prices']) / len(df4444))
    df2222['frauds'] = fraud_level
    df2222['fraud_share'] = df2222['frauds'] / df2222['count']
    df2222['prices'] = prices
    df2222 = df2222.sort_values(by='capacity', ascending=False)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df2222, y="fraud_share", x="capacity", color='#6693F5')
    plt.title("Fraud by capacity")

    df11 = mostdetaillproductclass#.loc[mostdetaillproductclass['product_model'] == 'iphone 13 pro max']
    ttest, pval = ttest_ind(df11.loc[df11['fraud'] == 1]['prices'].values.tolist(), df11.loc[df11['fraud'] == 0]['prices'].values.tolist(), equal_var=False)
    print('p value', pval)
    a = sns.FacetGrid(df11, hue='fraud', aspect=2)
    a.map(sns.kdeplot, 'prices', shade=True)
    a.set(xlim=(0, df11['prices'].max()))
    #plt.title("Iphone 13 Pro Max: Distribution by Price")
    plt.title("All products: Distribution by price")
    a.add_legend()
    plt.show()



    del mostdetaillproductclass['prices'], mostdetaillproductclass['shippings'], mostdetaillproductclass['user-ID'], mostdetaillproductclass['Sicher_bezahlen']
    del mostdetaillproductclass['Gewerblicher_user'], mostdetaillproductclass['offeringsonline'], mostdetaillproductclass['offeringssum'], mostdetaillproductclass['profilerating']
    del mostdetaillproductclass['profilefriendliness'], mostdetaillproductclass['profilereliability'], mostdetaillproductclass['profilereplyrate']
    del mostdetaillproductclass['profilereplyspeed'], mostdetaillproductclass['profilefollowers'], mostdetaillproductclass['startyear']
    del mostdetaillproductclass['startweekday'], mostdetaillproductclass['startmonth'], mostdetaillproductclass['startdayofmonth']
    del mostdetaillproductclass['index'], mostdetaillproductclass['LAT'], mostdetaillproductclass['LNG'], mostdetaillproductclass['PLZ']

    # STATS
    df_statistics = pd.DataFrame()
    df_statistics['mean'] = np.around(mostdetaillproductclass.mean(), 3)
    df_statistics['mean'] = np.around(df_statistics['mean'], 3)
    df_statistics['median'] = np.around(mostdetaillproductclass.median(), 3)
    df_statistics['median'] = np.around(df_statistics['median'], 3)
    df_statistics['max'] = mostdetaillproductclass.max()
    df_statistics['min'] = mostdetaillproductclass.min()
    df_statistics['count'] = mostdetaillproductclass.count()
    df_statistics['std'] = np.around(mostdetaillproductclass.std(), 3)
    df_statistics['std'] = np.around(df_statistics['std'], 3)


    # slope, intercept, r_value, p_value, std_err = stats.linregress(df['columnxy'], df['fraud'])
    slope = []
    r_value = []
    p_value = []
    for i in mostdetaillproductclass.columns:
        if mostdetaillproductclass[str(i)].dtypes == int or mostdetaillproductclass[str(i)].dtypes == float:
            c = stats.linregress(mostdetaillproductclass[str(i)], mostdetaillproductclass['fraud'])
            slope.append("{:.5f}".format(float(c[0])))
            r_value.append("{:.5f}".format(float(c[2])))
            p_value.append("{:.5f}".format(float(c[3])))
    df_statistics['slope'] = slope
    df_statistics['r_value'] = r_value
    df_statistics['p_value'] = p_value
    print(df_statistics)
    with open('stats_products.textmate', 'w') as file:
        file.write(str(df_statistics) + '\n')
    del mostdetaillproductclass['offer-ID']

    mostdetaillproductclass.rename(columns={'iphone 14 (no pro max)': '14 (no pro max)', 'iphone 13 (no pro max)': '13 (no pro max)',
                         'iphone 13 pro max': '13 pro max', 'iphone 14 pro max': '14 pro max'}, inplace=True)

    corr = mostdetaillproductclass.corr(method='pearson')
    plt.figure(figsize=(20, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm_r', vmin=-0.4, vmax=0.3, fmt='.2f')
    plt.title('Correlation matrix')
    plt.xticks(rotation=40)
    plt.show()


def titles_plot(titles_analysed):
    del titles_analysed['offer-ID']

    # STATS
    df_statistics = pd.DataFrame()
    df_statistics['mean'] = np.around(titles_analysed.mean(), 3)
    df_statistics['mean'] = np.around(df_statistics['mean'], 3)
    df_statistics['median'] = np.around(titles_analysed.median(), 3)
    df_statistics['median'] = np.around(df_statistics['median'], 3)
    df_statistics['max'] = titles_analysed.max()
    df_statistics['min'] = titles_analysed.min()
    df_statistics['count'] = titles_analysed.count()
    df_statistics['std'] = np.around(titles_analysed.std(), 3)
    df_statistics['std'] = np.around(df_statistics['std'], 3)

    # slope, intercept, r_value, p_value, std_err = stats.linregress(df['columnxy'], df['fraud'])
    slope = []
    r_value = []
    p_value = []
    for i in titles_analysed.columns:
        if titles_analysed[str(i)].dtypes == int or titles_analysed[str(i)].dtypes == float:
            c = stats.linregress(titles_analysed[str(i)], titles_analysed['fraud'])
            slope.append("{:.5f}".format(float(c[0])))
            r_value.append("{:.5f}".format(float(c[2])))
            p_value.append("{:.5f}".format(float(c[3])))
    df_statistics['slope'] = slope
    df_statistics['r_value'] = r_value
    df_statistics['p_value'] = p_value
    print(df_statistics)
    with open('stats_titles.textmate', 'w') as file:
        file.write(str(df_statistics) + '\n')

    # PLOTS

    corr = titles_analysed.corr(method='pearson')
    plt.figure(figsize=(20, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm_r', vmin=-0.6, vmax=0.6, fmt='.2f')
    plt.title('Correlation matrix')
    plt.xticks(rotation=20)
    plt.show()


def descriptions1_plot():
    descriptions_analysed = pd.read_csv('csv_data/descriptions_copy.csv')
    del descriptions_analysed['offer-ID']

    # STATS
    df_statistics = pd.DataFrame()
    df_statistics['mean'] = np.around(descriptions_analysed.mean(), 3)
    df_statistics['mean'] = np.around(df_statistics['mean'], 3)
    df_statistics['median'] = np.around(descriptions_analysed.median(), 3)
    df_statistics['median'] = np.around(df_statistics['median'], 3)
    df_statistics['max'] = descriptions_analysed.max()
    df_statistics['min'] = descriptions_analysed.min()
    df_statistics['count'] = descriptions_analysed.count()
    df_statistics['std'] = np.around(descriptions_analysed.std(), 3)
    df_statistics['std'] = np.around(df_statistics['std'], 3)

    # slope, intercept, r_value, p_value, std_err = stats.linregress(df['columnxy'], df['fraud'])
    slope = []
    r_value = []
    p_value = []
    for i in descriptions_analysed.columns:
        if descriptions_analysed[str(i)].dtypes == int or descriptions_analysed[str(i)].dtypes == float:
            c = stats.linregress(descriptions_analysed[str(i)], descriptions_analysed['fraud'])
            slope.append("{:.5f}".format(float(c[0])))
            r_value.append("{:.5f}".format(float(c[2])))
            p_value.append("{:.5f}".format(float(c[3])))
    df_statistics['slope'] = slope
    df_statistics['r_value'] = r_value
    df_statistics['p_value'] = p_value
    print(df_statistics)
    with open('stats_descriptions.textmate', 'w') as file:
        file.write(str(df_statistics) + '\n')

    # PLOTS
    corr = descriptions_analysed.corr(method='pearson')
    plt.figure(figsize=(20, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm_r', vmin=-0.5, vmax=0.7, fmt='.2f')
    plt.title('Correlation matrix')
    plt.xticks(rotation=30)
    plt.show()


def pictures_plot(apple_offers):

    old_data = pd.read_csv('csv_data/reverse_image_search.csv')
    del old_data['fraud'], old_data['offer-ID']
    old_data['pics'] = old_data['pics'].astype(str)
    new_list = []
    for i in old_data['pics']:
        new_list.append(i[:75])
    old_data['pic_short'] = new_list

    apple_offers2 = pd.DataFrame()
    apple_offers2['pics'] = apple_offers['pics']
    apple_offers2['pics'] = apple_offers2['pics'].astype(str)
    apple_offers2['pics'] = apple_offers['pics']
    apple_offers2['fraud'] = apple_offers['fraud']
    apple_offers2['offer-ID'] = apple_offers['offer-ID']
    new_list = []
    for i in apple_offers2['pics']:
        new_list.append(i[:75])
    apple_offers2['pic_short'] = new_list

    pictures_data = pd.merge(old_data, apple_offers2, on='pic_short', how='left')
    del pictures_data['pics_y'], pictures_data['pic_short']
    pictures_data = pictures_data.loc[pictures_data['similar_images'] != 1]

    del pictures_data['offer-ID']

    # STATS
    df_statistics = pd.DataFrame()
    df_statistics['mean'] = np.around(pictures_data.mean(), 3)
    df_statistics['mean'] = np.around(df_statistics['mean'], 3)
    df_statistics['median'] = np.around(pictures_data.median(), 3)
    df_statistics['median'] = np.around(df_statistics['median'], 3)
    df_statistics['max'] = pictures_data.max()
    df_statistics['min'] = pictures_data.min()
    df_statistics['count'] = pictures_data.count()
    df_statistics['std'] = np.around(pictures_data.std(), 3)
    df_statistics['std'] = np.around(df_statistics['std'], 3)

    # slope, intercept, r_value, p_value, std_err = stats.linregress(df['columnxy'], df['fraud'])
    slope = []
    r_value = []
    p_value = []
    for i in pictures_data.columns:
        if pictures_data[str(i)].dtypes == int or pictures_data[str(i)].dtypes == float:
            c = stats.linregress(pictures_data[str(i)], pictures_data['fraud'])
            slope.append("{:.5f}".format(float(c[0])))
            r_value.append("{:.5f}".format(float(c[2])))
            p_value.append("{:.5f}".format(float(c[3])))
    df_statistics['slope'] = slope
    df_statistics['r_value'] = r_value
    df_statistics['p_value'] = p_value
    print(df_statistics)
    with open('stats_pictures.textmate', 'w') as file:
        file.write(str(df_statistics) + '\n')

    pictures_data.rename(columns={'pages_with_matching_images': 'pages'}, inplace=True)
    # PLOTS
    corr = pictures_data.corr(method='pearson')
    plt.figure(figsize=(3, 3))
    sns.heatmap(corr, annot=True, cmap='coolwarm_r', vmin=-0.3, vmax=0.3, fmt='.2f')
    plt.title('Correlation matrix')
    plt.xticks(rotation=10)
    plt.show()


def location_plot(location_analysed):
    del location_analysed['west'], location_analysed['PLZ'], location_analysed['offer-ID'], location_analysed['qkm']

    # BUNDESLAND COUNT
    df2 = location_analysed
    df2 = df2['BUNDESLAND'].value_counts().to_frame()
    df2 = df2.reset_index()
    df2.columns = ['BUNDESLAND', 'count']
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df2, y='BUNDESLAND', x='count', color='orange')
    plt.title("BUNDESLAND count")


    # FRAUD SHARE
    df3 = location_analysed
    fraud_level = []
    prices = []
    for i in df2['BUNDESLAND']:
        df4 = df3.loc[df3['BUNDESLAND'] == str(i)]
        fraud_level.append(sum(df4['fraud']))
        prices.append(sum(df4['prices']) / len(df4))
    df2['frauds'] = fraud_level
    df2['fraud_share'] = df2['frauds'] / df2['count']
    df2['prices'] = prices
    df2 = df2.sort_values(by='fraud_share', ascending=False)
    print(df2)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df2, x="fraud_share", y="BUNDESLAND", color='orange')
    plt.title("Fraud by BUNDESLAND")


    # Thüringen: FRAUD SHARE
    df222 = location_analysed.loc[(location_analysed['BUNDESLAND'] == 'Hessen')]
    df222 = df222['STADT'].value_counts().to_frame()
    df222 = df222.reset_index()
    df222.columns = ['STADT', 'count']
    df333 = location_analysed.loc[(location_analysed['BUNDESLAND'] == 'Hessen')]
    fraud_level = []
    prices = []
    for i in df222['STADT']:
        df444 = df333.loc[df333['STADT'] == str(i)]
        fraud_level.append(sum(df444['fraud']))
        prices.append(sum(df444['prices']) / len(df444))
    df222['frauds'] = fraud_level
    df222['fraud_share'] = df222['frauds'] / df222['count']
    df222['prices'] = prices
    df222 = df222.sort_values(by='fraud_share', ascending=False)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df222, x="fraud_share", y="STADT", color='orange')
    plt.title("Fraud by STADT (Thüringen)")


    # PRICE LEVELS
    df2 = df2.sort_values(by='prices', ascending=False)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df2, x="prices", y="BUNDESLAND", color='orange')
    plt.title("Mean prices by BUNDESLAND")


    df11 = location_analysed
    a = sns.FacetGrid(df11, hue='fraud', aspect=4)
    a.map(sns.kdeplot, 'LNG', shade=True)
    a.set(xlim=(0, df11['LNG'].max()))
    plt.title("Iphone 13 Pro Max: Distribution by Price")
    a.add_legend()


    '''

    # FOLIUM
    m = folium.Map(location=[50.32, 9.5], tiles='openstreetmap', zoom_start=6)
    df3 = df1[['LAT', 'LNG', 'fraud_per_capita']].values.tolist()
    #HeatMap(data=df3, radius=15).add_to(m)
    f = open('/Users/philippschenk/PycharmProjects/ebay_scraper2/csv_data/plz-1stellig.geojson')
    cities = json.load(f)
    folium.GeoJson(cities, name="geojson").add_to(m)
    #folium.TopoJson(json.loads(requests.get(antarctic_ice_shelf_topo).text), "objects.antarctic_ice_shelf", name="topojson").add_to(m)
    #folium.Choropleth(geo_data=df1, data=cities, columns=["PLZ", "Unemployment"], key_on="feature.plz", fill_color="BuPu", fill_opacity=0.7,
    #                line_opacity=0.5, legend_name="Unemployment Rate (%)", reset=True).add_to(m)
    folium.LayerControl().add_to(m)
    m.save('output_file.html')

    # PLOTLY
    plz_1 = []
    for i in df1['PLZ']:
        i = str(i)
        plz_1.append((0))#i[0]))
    df1['plz'] = plz_1
    df1['plz'] = df1['plz'].astype(int)
    df1 = df1.reset_index()
    del df1['index']
    f = open('/Users/philippschenk/PycharmProjects/ebay_scraper2/csv_data/plz-1stellig.geojson')
    cities = json.load(f)

    #f = open('/Users/philippschenk/PycharmProjects/ebay_scraper2/tr-cities-utf8.json')
    #cities = json.load(f)
    print(cities["features"][0]["properties"]['plz'])
    print(cities["features"][0]["geometry"]['coordinates'])
    #geojson = cities["features"][0]["geometry"]['coordinates']
    print(df1.head())

    fig = px.choropleth_mapbox(df1, geojson=cities["features"], locations=df1.plz,  #df1['plz']
                               #featureidkey=cities["features"][0]["properties"]['plz'],
                               color=df1["einwohner"], hover_name="BUNDESLAND",
                               color_continuous_scale="Viridis",
                               mapbox_style="carto-positron",
                               zoom=5.5, center={"lat": 50.9, "lon": 9.45},
                               opacity=0.7)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    #fig.show()
    '''

    #del location_analysed['BUNDESLAND'], location_analysed['STADT']

    # STATS
    df_statistics = pd.DataFrame()
    df_statistics['mean'] = np.around(location_analysed.mean(), 3)
    df_statistics['mean'] = np.around(df_statistics['mean'], 3)
    df_statistics['median'] = np.around(location_analysed.median(), 3)
    df_statistics['median'] = np.around(df_statistics['median'], 3)
    df_statistics['max'] = location_analysed.max()
    df_statistics['min'] = location_analysed.min()
    df_statistics['count'] = location_analysed.count()
    df_statistics['std'] = np.around(location_analysed.std(), 3)
    df_statistics['std'] = np.around(df_statistics['std'], 3)
    print(location_analysed.head())
    print(df_statistics)

    # slope, intercept, r_value, p_value, std_err = stats.linregress(df['columnxy'], df['fraud'])
    slope = []
    r_value = []
    p_value = []
    for i in location_analysed.columns:
        if location_analysed[str(i)].dtypes == int or location_analysed[str(i)].dtypes == float:
            c = stats.linregress(location_analysed[str(i)], location_analysed['fraud'])
            slope.append("{:.5f}".format(float(c[0])))
            r_value.append("{:.5f}".format(float(c[2])))
            p_value.append("{:.5f}".format(float(c[3])))
    df_statistics['slope'] = slope
    df_statistics['r_value'] = r_value
    df_statistics['p_value'] = p_value
    print(df_statistics)
    with open('stats_locations.textmate', 'w') as file:
        file.write(str(df_statistics) + '\n')
    #del location_analysed['offer-ID']

    # PLOTS
    corr = location_analysed.corr(method='pearson')
    plt.figure(figsize=(20, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm_r', vmin=-0.4, vmax=0.6, fmt='.2f')
    plt.title('Correlation matrix')
    plt.xticks(rotation=40)
    #plt.show()

    df11 = location_analysed.loc[(location_analysed['BUNDESLAND'] == 'Hamburg') | (location_analysed['BUNDESLAND'] == 'Rheinland-Pfalz')].reset_index(drop=True)
    ttest, pval = ttest_ind(df11.loc[df11['BUNDESLAND'] == 'Hamburg']['prices'].values.tolist(), df11.loc[df11['BUNDESLAND'] == 'Rheinland-Pfalz']['prices'].values.tolist(), equal_var=False)
    print('p value', pval)
    a = sns.FacetGrid(df11, hue='BUNDESLAND', aspect=2)
    a.map(sns.kdeplot, 'prices', shade=True)
    a.set(xlim=(0, df11['prices'].max()))
    # plt.title("Iphone 13 Pro Max: Distribution by Price")
    plt.title("All products: Distribution by price")
    a.add_legend()
    plt.show()


def time_scraping_plot(apple_offers):

    df2 = apple_offers
    day = []
    month = []
    year = []
    for i in df2['dates']:
        i = i.split('.')
        day.append(i[0])
        month.append(i[1])
        year.append(i[2])
    df2['day'] = day
    df2['day'] = df2['day'].astype(int)
    df2['month'] = month
    df2['month'] = df2['month'].astype(int)
    df2['year'] = year
    df2['year'] = df2['year'].astype(int)
    df2 = df2.loc[(df2['day'] > 2) & (df2['month'] > 11)]
    df2 = df2.sort_values(by='dates', ascending=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(data=df2, x="dates", color='#6693F5', binwidth=1)
    ax.tick_params(axis='x', rotation=45)
    plt.title("Scraped offers by date")
    plt.show()


    df3 = pd.DataFrame()
    df3['startyear'] = df2['startyear']
    df3['startweekday'] = df2['startweekday']
    df3['startmonth'] = df2['startmonth']
    df3['startdayofmonth'] = df2['startdayofmonth']
    df3['offer_day'] = df2['day']
    df3['offer-ID'] = df2['offer-ID']
    df3['fraud'] = df2['fraud']

    del df3['offer-ID']

    # STATS
    df_statistics = pd.DataFrame()
    df_statistics['mean'] = np.around(df3.mean(), 3)
    df_statistics['mean'] = np.around(df_statistics['mean'], 3)
    df_statistics['median'] = np.around(df3.median(), 3)
    df_statistics['median'] = np.around(df_statistics['median'], 3)
    df_statistics['max'] = df3.max()
    df_statistics['min'] = df3.min()
    df_statistics['count'] = df3.count()
    df_statistics['std'] = np.around(df3.std(), 3)
    df_statistics['std'] = np.around(df_statistics['std'], 3)

    # slope, intercept, r_value, p_value, std_err = stats.linregress(df['columnxy'], df['fraud'])
    slope = []
    r_value = []
    p_value = []
    for i in df3.columns:
        if df3[str(i)].dtypes == int or df3[str(i)].dtypes == float:
            c = stats.linregress(df3[str(i)], df3['fraud'])
            slope.append("{:.5f}".format(float(c[0])))
            r_value.append("{:.5f}".format(float(c[2])))
            p_value.append("{:.5f}".format(float(c[3])))
    df_statistics['slope'] = slope
    df_statistics['r_value'] = r_value
    df_statistics['p_value'] = p_value
    print(df_statistics)
    with open('stats_timing.textmate', 'w') as file:
        file.write(str(df_statistics) + '\n')

    df3.rename(columns={'startdayofmonth': 'dayofmonth'}, inplace=True)
    # PLOTS
    corr = df3.corr(method='pearson')
    plt.figure(figsize=(5, 3))
    sns.heatmap(corr, annot=True, cmap='coolwarm_r', vmin=-0.2, vmax=0.25, fmt='.2f')
    plt.title('Correlation matrix')
    plt.xticks(rotation=20)
    plt.show()


def profile_plot(profile_analysed):
    print(profile_analysed.head(1))
    del profile_analysed['prices']
    del profile_analysed['offer-ID']

    # STATS
    df_statistics = pd.DataFrame()
    df_statistics['mean'] = np.around(profile_analysed.mean(), 3)
    df_statistics['mean'] = np.around(df_statistics['mean'], 3)
    df_statistics['median'] = np.around(profile_analysed.median(), 3)
    df_statistics['median'] = np.around(df_statistics['median'], 3)
    df_statistics['max'] = profile_analysed.max()
    df_statistics['min'] = profile_analysed.min()
    df_statistics['count'] = profile_analysed.count()
    df_statistics['std'] = np.around(profile_analysed.std(), 3)
    df_statistics['std'] = np.around(df_statistics['std'], 3)

    # slope, intercept, r_value, p_value, std_err = stats.linregress(df['columnxy'], df['fraud'])
    slope = []
    r_value = []
    p_value = []
    for i in profile_analysed.columns:
        if profile_analysed[str(i)].dtypes == int or profile_analysed[str(i)].dtypes == float:
            c = stats.linregress(profile_analysed[str(i)], profile_analysed['fraud'])
            slope.append("{:.5f}".format(float(c[0])))
            r_value.append("{:.5f}".format(float(c[2])))
            p_value.append("{:.5f}".format(float(c[3])))
    df_statistics['slope'] = slope
    df_statistics['r_value'] = r_value
    df_statistics['p_value'] = p_value
    print(df_statistics)
    with open('stats_profiles.textmate', 'w') as file:
        file.write(str(df_statistics) + '\n')

    # PLOTS
    corr = profile_analysed.corr(method='pearson')
    plt.figure(figsize=(20, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm_r', vmin=-0.5, vmax=0.8, fmt='.2f')
    plt.title('Correlation matrix')
    plt.xticks(rotation=35)
    plt.show()


main()