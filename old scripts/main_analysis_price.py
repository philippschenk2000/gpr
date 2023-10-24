import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from scipy.stats import ttest_ind
from simpledbf import Dbf5
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split  # to split the data
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
plt.rcParams["font.family"] = "Times New Roman"


def main():

    apple_offers = pd.read_csv('csv_data/profiles_offers_apple_clean.csv')
    #apple_offers = pd.read_csv('csv_data/profiles_offers_clean.csv')
    checked_results = pd.read_csv('csv_data/checked_results.csv')
    checked_results = checked_results.loc[checked_results['fraud'] == 0].reset_index(drop=True)
    apple_offers = pd.merge(apple_offers, checked_results, on='offer-ID', how='left')
    del apple_offers['views'], apple_offers['name'], apple_offers['apple_url'], apple_offers['apple-offer-url'], apple_offers['scrape_time'], apple_offers['fraud']


    ''' APPLE OFFERS ONLY - EXTENDING DATASET '''
    for i in range(0, 1):
        apple_offers = latlong(apple_offers)                                    # Location information (Berlin Kreuzberg, Berlin,  52.5323,  13.3846)
        apple_offers['PLZ'] = apple_offers['PLZ'].astype(int)
        apple_offers = apple_offers.loc[(apple_offers['prices'] > 100) & (apple_offers['prices'] < 1900) & (apple_offers['PLZ'] < 99999)].reset_index(drop=True)
        mainproductclass = product_classifiing1(apple_offers)                   # Product class (iphone)
        detaillproductclass = product_classifiing2(mainproductclass)            # Exact product model (iphone 12 pro max)
        mostdetaillproductclass = product_classifiing3(detaillproductclass)     # Capacity (128)
        location_analysed = location_analysis(apple_offers)                     # Relationship with lat/lng/plz/bundesland and fraud
        times = times_analysis(apple_offers)                     # Relationship with lat/lng/plz/bundesland and fraud

        final_data = pd.merge(mostdetaillproductclass, location_analysed, on='offer-ID', how='left')
        final_data = final_data.drop_duplicates(subset=['offer-ID'], keep='first')
        final_data.rename(columns={'PLZ_y': 'PLZ', 32.0: '32GB', 64.0: '64GB', 128.0: '128GB', 256.0: '256GB', 512.0: '512GB'}, inplace=True)
        final_data = final_data.loc[(final_data['prices'] > 200) & (final_data['others'] == 0) & (final_data['capacity'] > 0)]
        del final_data['PLZ_x'], final_data['32GB'], final_data['NUTS1']
        with open('final_dataset.textmate', 'w') as file:
            file.write(str(final_data) + '\n')


        ''' EXPLORATIVE ANALYSIS '''
        for i in range(0, 1):
            product_classes = product_classes_plot(final_data, times)

            for i in range(0, 1):
                for i in range(0, 1):
                    del final_data['offer-ID'], final_data['capacity']
                    fig, ax = plt.subplots(figsize=(15, 12))
                    corr = final_data.corr()
                    custom_palette = sns.color_palette("coolwarm_r", 24, as_cmap=True)
                    sns.heatmap(corr, cmap=custom_palette, annot_kws={'size': 20}, vmin=-0.6, vmax=0.6)
                    plt.title("Correlation Matrix \n (use for reference)", fontsize=14)
                    #plt.show()





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

    df1 = df1.sort_values('PLZ', ascending=True)
    df1.fillna(method='ffill', inplace=True)
    df1.fillna(method='bfill', inplace=True)
    df1 = df1.loc[(df1['prices'] > 50) & (df1['prices'] < 2800) & (df1['PLZ'] < 99999)]
    df1 = df1.reset_index()

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
    X_cat['others'] = X_cat['other'] + X_cat['pencil'] + X_cat['tv'] + X_cat['imac'] + X_cat['keyboard'] + X_cat['mac mini'] + X_cat['macbook'] + X_cat['ipad'] + X_cat['macbook air'] + X_cat['macbook air 2019'] + X_cat['macbook air 2020'] + X_cat['macbook air 2021'] + X_cat['macbook air 2022'] + X_cat['macbook pro'] + X_cat['macbook pro 2019'] + X_cat['macbook pro 2020'] + X_cat['macbook pro 2021'] + X_cat['macbook pro 2022'] + X_cat['iphone x'] + X_cat['iphone xr'] + X_cat['iphone xs'] + X_cat['iphone 7'] + X_cat['iphone 7 plus'] + X_cat['iphone 8'] + X_cat['iphone 8 plus'] + X_cat['iphone'] + X_cat['watch'] + X_cat['watch se'] + X_cat['watch series'] + X_cat['watch ultra'] + X_cat['ipad pro'] + X_cat['ipad mini'] + X_cat['ipad air']
    del X_cat['pencil'], X_cat['other'], X_cat['tv'], X_cat['imac'], X_cat['keyboard'], X_cat['mac mini'], X_cat['macbook'], X_cat['ipad'], X_cat['macbook air'], X_cat['macbook air 2019'], X_cat['macbook air 2020'], X_cat['macbook air 2021'], X_cat['macbook air 2022']
    del X_cat['macbook pro 2019'], X_cat['macbook pro 2020'], X_cat['macbook pro 2021'], X_cat['macbook pro 2022'], X_cat['macbook pro']
    del X_cat['watch se'], X_cat['watch series'], X_cat['watch ultra'], X_cat['watch']
    del X_cat['iphone x'], X_cat['iphone xr'], X_cat['iphone xs'], X_cat['iphone 7'], X_cat['iphone 7 plus'], X_cat['iphone 8'], X_cat['iphone 8 plus'], X_cat['iphone']
    del X_cat['airpods 2'], X_cat['airpods 3']
    del X_cat['airpods 2 pro'], X_cat['airpods max']
    del X_cat['ipad air'], X_cat['ipad mini'], X_cat['ipad pro']
    X_cat['offer-ID'] = mainproductclass['offer-ID']
    mainproductclass = pd.merge(mainproductclass, X_cat, on='offer-ID', how='left')
    mainproductclass['iphone 11'] = mainproductclass['iphone 11'].astype(int)
    mainproductclass['iphone 11 pro'] = mainproductclass['iphone 11 pro'].astype(int)
    mainproductclass['iphone 11 pro max'] = mainproductclass['iphone 11 pro max'].astype(int)
    mainproductclass['iphone 12'] = mainproductclass['iphone 12'].astype(int)
    mainproductclass['iphone 12 mini'] = mainproductclass['iphone 12 mini'].astype(int)
    mainproductclass['iphone 12 pro'] = mainproductclass['iphone 12 pro'].astype(int)
    mainproductclass['iphone 12 pro max'] = mainproductclass['iphone 12 pro max'].astype(int)
    mainproductclass['iphone 13'] = mainproductclass['iphone 13'].astype(int)
    mainproductclass['iphone 13 mini'] = mainproductclass['iphone 13 mini'].astype(int)
    mainproductclass['iphone 13 pro'] = mainproductclass['iphone 13 pro'].astype(int)
    mainproductclass['iphone 13 pro max'] = mainproductclass['iphone 13 pro max'].astype(int)
    mainproductclass['iphone 14'] = mainproductclass['iphone 14'].astype(int)
    mainproductclass['iphone 14 pro'] = mainproductclass['iphone 14 pro'].astype(int)
    mainproductclass['iphone 14 plus'] = mainproductclass['iphone 14 plus'].astype(int)
    mainproductclass['iphone 14 pro max'] = mainproductclass['iphone 14 pro max'].astype(int)
    mainproductclass['others'] = mainproductclass['others'].astype(int)

    return mainproductclass


def product_classifiing3(detaillproductclass):

    most_detailled_product_class = []
    for j in range(0, len(detaillproductclass['titles'])):
        i = detaillproductclass['titles'][j].lower()
        ii = detaillproductclass['descriptions'][j].lower()
        # IPHONES
        if 'phon' in i and '16' in i or 'phon' in i and '16 G' in ii or 'phon' in i and '16G' in ii:
            most_detailled_product_class.append(16)
        elif 'phon' in i and '32' in i or 'phon' in i and '32 G' in ii or 'phon' in i and '32G' in ii:
            most_detailled_product_class.append(32)
        elif 'phon' in i and '64' in i or 'phon' in i and '64' in ii:
            most_detailled_product_class.append(64)
        elif 'phon' in i and '128' in i or 'phon' in i and '128' in ii:
            most_detailled_product_class.append(128)
        elif 'phon' in i and '256' in i or 'phon' in i and '256' in ii:
            most_detailled_product_class.append(256)
        elif 'phon' in i and '512' in i or 'phon' in i and '512' in ii:
            most_detailled_product_class.append(512)
        elif 'phon' in i and '1TB' in i or 'phon' in i and '1TB' in ii:
            most_detailled_product_class.append(1000)
        elif 'phon' in i and '1 TB' in i or 'phon' in i and '1 TB' in ii:
            most_detailled_product_class.append(1000)
        elif 'phon' in i:
            most_detailled_product_class.append(None)

        # IPADS
        elif 'pad' in i and '16' in i or 'pad' in i and '16 G' in ii or 'pad' in i and '16G' in ii:
            most_detailled_product_class.append(16)
        elif 'pad' in i and '32' in i or 'pad' in i and '16 G' in ii or 'pad' in i and '16G' in ii:
            most_detailled_product_class.append(32)
        elif 'pad' in i and '64' in i or 'phon' in i and '64' in ii:
            most_detailled_product_class.append(64)
        elif 'pad' in i and '128' in i or 'phon' in i and '128' in ii:
            most_detailled_product_class.append(128)
        elif 'pad' in i and '256' in i or 'phon' in i and '256' in ii:
            most_detailled_product_class.append(256)
        elif 'pad' in i and '512' in i or 'phon' in i and '512' in ii:
            most_detailled_product_class.append(512)
        elif 'pad' in i:
            most_detailled_product_class.append(None)

        # MACBOOKS
        elif 'book' in i and '128' in i or 'phon' in i and '128' in ii:
            most_detailled_product_class.append(128)
        elif 'book' in i and '256' in i or 'book' in i and '256' in ii:
            most_detailled_product_class.append(256)
        elif 'book' in i and '512' in i or 'book' in i and '512' in ii:
            most_detailled_product_class.append(512)
        elif 'book' in i and '1TB' in i or 'book' in i and '1TB' in ii:
            most_detailled_product_class.append(1000)
        elif 'book' in i and '1 TB' in i or 'book' in i and '1 TB' in ii:
            most_detailled_product_class.append(1000)
        elif 'book' in i and '2TB' in i or 'book' in i and '2TB' in ii:
            most_detailled_product_class.append(2000)
        elif 'book' in i and '2 TB' in i or 'book' in i and '2 TB' in ii:
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
    del detaillproductclass['STADT'], detaillproductclass['BUNDESLAND'], detaillproductclass['LAT'], detaillproductclass['LNG'], detaillproductclass['prices']
    X_cat = pd.get_dummies(detaillproductclass['capacity'], drop_first=True)
    X_cat['offer-ID'] = detaillproductclass['offer-ID']
    detaillproductclass = pd.merge(detaillproductclass, X_cat, on='offer-ID', how='left')

    return detaillproductclass


def location_analysis(apple_offers):

    nuts2_de = pd.read_csv('csv_data/pc2020_DE_NUTS-2021_v1.0.csv', delimiter=';')
    pop_density = pd.read_csv('csv_data/bahn.csv')
    crime_rate_nuts3 = pd.read_csv('csv_data/crime_rate_nuts3.csv', delimiter=';')
    dbf2 = Dbf5('csv_data/plz-5stellig.dbf', codec='latin')
    df2 = dbf2.to_dataframe()

    df1 = pd.DataFrame()
    df1['offer-ID'] = apple_offers['offer-ID']
    df1['PLZ'] = apple_offers['PLZ'].astype(int)
    df1['STADT'] = apple_offers['STADT']
    df1['BUNDESLAND'] = apple_offers['BUNDESLAND']
    df1['LAT'] = apple_offers['LAT']
    df1['LNG'] = apple_offers['LNG']
    df1['prices'] = apple_offers['prices']
    nuts1 = []
    nuts2 = []
    nuts3 = []
    plz = []
    for x in nuts2_de['NUTS3']:
        nuts3.append(x[1:6])
        nuts2.append(x[1:5])
        nuts1.append(x[1:4])
    for x in nuts2_de['PLZ']:
        plz.append(x[1:-1])

    nuts2_de['NUTS3'] = nuts3
    nuts2_de['NUTS2'] = nuts2
    nuts2_de['NUTS1'] = nuts1
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
    df3['offer_day'] = df2['day'].astype(int)
    df3['offer_month'] = df2['month'].astype(int)
    df3['offer_year'] = df2['year'].astype(int)
    df3['date'] = df2['dates']
    df3['offer-ID'] = df2['offer-ID']
    return df3




def product_classes_plot(final_data, times):
    for i in range(0, 1):
        del final_data['shippings'], final_data['user-ID'], final_data['Sicher_bezahlen'], final_data['scraptime'], final_data['product']
        del final_data['Gewerblicher_user'], final_data['offeringsonline'], final_data['offeringssum'], final_data['profilerating']
        del final_data['profilefriendliness'], final_data['profilereliability'], final_data['profilereplyrate']
        del final_data['profilereplyspeed'], final_data['profilefollowers'], final_data['startyear']
        del final_data['startweekday'], final_data['startmonth'], final_data['startdayofmonth']
        del final_data['pics'], final_data['titles'], final_data['descriptions'], final_data['index']
        del final_data['others'], final_data['dates']
        with open('final_dataset.textmate', 'w') as file:
            file.write(str(final_data) + '\n')


    ''' PRODUCTS '''
    products = final_data.copy()
    for i in range(0, 1):
        break
        # PRICE LEVELS
        df2 = products.sort_values(by='prices', ascending=False)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=df2, x="prices", y="product_model", color='#6693F5')
        plt.title("Mean prices by product")
    predicted_prices = predict_price(products)
    predicted_prices = predicted_prices.loc[(predicted_prices['cheaper'] > -26) & (predicted_prices['cheaper'] < 26)]
    for i in range(0, 1):
        new_df = pd.merge(final_data, predicted_prices, on='offer-ID', how='left')
        new_df = new_df.loc[new_df['cheaper'] > 0]
        new_df['cheaper'] = new_df['cheaper'] - sum(new_df['cheaper'])/len(new_df['cheaper'])
        del new_df['STADT'], new_df['BUNDESLAND'], new_df['LAT'], new_df['LNG'], new_df['PLZ'], new_df['NUTS3'], new_df['NUTS2'], new_df['pop_dichte'], new_df['crime_rate']
        del new_df['einwohner'], new_df['qkm'], new_df['east'], new_df['west']
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(data=new_df, y='product_model', x='cheaper', color='orange')
        plt.title("price difference per product")


    ''' LOCATIONS '''
    locations = pd.merge(final_data, predicted_prices, on='offer-ID', how='left')
    locations = locations.drop_duplicates(subset=['offer-ID'], keep='last')
    del locations['y_test'], locations['iphone 11'], locations['iphone 11 pro'], locations['iphone 11 pro max'], locations['iphone 12'], locations['iphone 12 mini'], locations['iphone 12 pro'], locations['iphone 12 pro max'], locations['west']
    del locations['iphone 13'], locations['iphone 13 mini'], locations['iphone 13 pro'], locations['iphone 13 pro max'], locations['iphone 14'], locations['iphone 14 plus'], locations['iphone 14 pro'], locations['iphone 14 pro max']
    del locations['y_pred_prob'], locations['prices'], locations['capacity']
    locations = locations.loc[(locations['cheaper'] > -26) & (locations['cheaper'] < 26)]
    locations['cheaper'] = locations['cheaper'] - sum(locations['cheaper'])/len(locations['cheaper'])
    with open('final_dataset.textmate', 'w') as file:
        file.write(str(locations) + '\n')
    print(locations.head())
    print(len(locations))

    further_research = 'BUNDESLAND'
    for i in range(0, 1):
        # BUNDESLAND COUNT
        df2 = locations
        df2 = df2[further_research].value_counts().to_frame()
        df2 = df2.reset_index()
        df2.columns = [further_research, 'count']
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(data=df2, y=further_research, x='count', color='orange')
        plt.title(further_research + " count")

        # PRICE LEVELS
        df2 = locations.sort_values(by='cheaper', ascending=False)
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(data=df2, x="cheaper", y=further_research, color='orange')
        plt.title("Mean price differences by " + further_research)

        break
        df11 = locations.loc[locations['east'] == 1]
        a = sns.FacetGrid(df11, hue=further_research, aspect=20, size=5)
        a.map(sns.kdeplot, 'cheaper', shade=True)
        a.set(xlim=(-30, 30))
        plt.title(further_research + ": Distribution by Price")
        a.add_legend()
    df_differences = pd.DataFrame()
    for i in range(0, 1):
        state1 = []
        state2 = []
        p_val = []
        for j in locations[further_research].unique():
            for l in locations[further_research].unique():
                ttest, pval = ttest_ind(locations.loc[locations[further_research] == j]['cheaper'].values.tolist(), locations.loc[locations[further_research] == l]['cheaper'].values.tolist(), equal_var=False)
                state1.append(j)
                state2.append(l)
                p_val.append(pval)
        df_differences['region1'] = state1
        df_differences['region2'] = state2
        df_differences['p_val'] = p_val
        df_differences['combi'] = df_differences['region1'] + ' & ' + df_differences['region2']
        df_differences = df_differences.sort_values(['p_val'], ascending=True)
        df_differences = df_differences.drop_duplicates(subset=['p_val'], keep='first')
        with open('geo_differences.textmate', 'w') as file:
            file.write(str(df_differences) + '\n')
        df_huge = df_differences.loc[df_differences['p_val'] < 0.08]

        # BUNDESLAND COUNT
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(data=df_huge, y='combi', x='p_val', color='orange')
        plt.title(further_research + " combinations: significant price differences")
        break

        # BUNDESLAND COUNT
        df2 = df_huge
        df2 = df2['region1'].value_counts().to_frame()
        df2 = df2.reset_index()
        df2.columns = ['region1', 'count']
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(data=df2, y='region1', x='count', color='orange')
        plt.title(further_research + " count: most significant price differences with other states")
        #plt.show()
    # give them ranges & then dummies; einwohner, qkm, usw.


    ''' TIMINGS '''
    timings = pd.merge(locations, times, on='offer-ID', how='left')
    timings = timings.loc[(timings['offer_month'] != 11) & (timings['offer_month'] != 10) & (timings['date'] != '01.04.2022') & (timings['date'] != '08.02.2022') & (timings['date'] != '01.12.2022') & (timings['date'] != '02.12.2022') & (timings['date'] != '03.12.2022') & (timings['date'] != '04.12.2022') & (timings['date'] != '05.12.2022') & (timings['date'] != '15.12.2022') & (timings['date'] != '16.12.2022') & (timings['date'] != '17.12.2022') & (timings['date'] != '18.12.2022') & (timings['date'] != '25.12.2022') & (timings['date'] != '06.01.2023') & (timings['date'] != '30.01.2023') & (timings['date'] != '11.02.2023') & (timings['date'] != '16.02.2023') & (timings['date'] != '17.02.2023') & (timings['date'] != '18.02.2023') & (timings['date'] != '19.02.2023') & (timings['date'] != '20.02.2023') & (timings['date'] != '24.02.2023')]
    del timings['PLZ'], timings['LAT'], timings['LNG'], timings['pop_dichte'], timings['crime_rate'], timings['einwohner'], timings['qkm'], timings['east']
    del timings['product_model'], timings['STADT'], timings['BUNDESLAND'], timings['NUTS3'], timings['NUTS2']
    timings = timings.sort_values(['offer_year', 'offer_month', 'offer_day'], ascending=True)

    further_research = 'date'
    for i in range(0, 1):
        df2 = timings
        df2 = df2[further_research].value_counts().to_frame()
        df2 = df2.reset_index()
        print(df2)

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(data=timings, y='cheaper', x=further_research, color='orange')
        plt.title("price difference over time")
        plt.xticks(rotation=40)
        plt.show()
    # start of month (gehalt); feiertag; weekday


    ''' PROFILE INFORMATION '''
    locations = pd.merge(final_data, predicted_prices, on='offer-ID', how='left')
    locations = locations.drop_duplicates(subset=['offer-ID'], keep='last')













def predict_price(products):

    df_statistics = pd.DataFrame()

    for i in range(0, 1):
        del products['STADT'], products['BUNDESLAND'], products['LAT'], products['LNG'], products['PLZ']
        del products['NUTS3'], products['NUTS2'], products['pop_dichte'], products['crime_rate']
        del products['einwohner'], products['qkm'], products['east'], products['west'], products['product_model'], products['capacity']
    # STATS
    # del products['offer-ID']
    df_statistics['mean'] = np.around(products.mean(), 3)
    df_statistics['mean'] = np.around(df_statistics['mean'], 3)
    df_statistics['max'] = products.max()
    df_statistics['min'] = products.min()
    df_statistics['count'] = products.count()
    slope = []
    r_value = []
    p_value = []
    for i in products.columns:
        #if products[str(i)].dtypes == int or products[str(i)].dtypes == float:
            c = stats.linregress(products[str(i)], products['prices'])
            slope.append("{:.5f}".format(float(c[0])))
            r_value.append("{:.5f}".format(float(c[2])))
            p_value.append("{:.5f}".format(float(c[3])))
        #else:
    df_statistics['slope'] = slope
    df_statistics['r_value'] = r_value
    df_statistics['p_value'] = p_value

    df_results = pd.DataFrame()
    for z in range(0, 1):
        #products2 = products.loc[products[str(z)] == 1].reset_index(drop=True)
        #print(len(products2))
        for y in range(0, 10):
            # 2 SPLITTING THE DATA & INPUTS
            products = products.sample(frac=1).reset_index(drop=True)
            X = products.drop('prices', axis=1)
            y = products['prices']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
            offer_ID = X_test['offer-ID'].reset_index(drop=True)
            X_test = X_test.drop('offer-ID', axis=1).reset_index(drop=True)
            X_train = X_train.drop('offer-ID', axis=1).reset_index(drop=True)
            # 4 MACHINE LEARNING ALGORITHMS
            MLA = [RandomForestRegressor()]
            X_test = X_test.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)
            df = pd.DataFrame()
            for j in range(0, len(MLA)):
                mla_now = MLA[j]
                mla_now.fit(X_train.values, y_train.values)
                y_pred_prob = mla_now.predict(X_test)
                feature = np.around(mla_now.feature_importances_, 4)
                feature = pd.Series(feature, X_train.columns)
                df_statistics[str(mla_now)[:30]] = feature
                df['y_pred_prob'] = y_pred_prob
                df['y_test'] = y_test
                df['offer-ID'] = offer_ID
                df['MLA'] = str(mla_now)[:10]
                df['cheaper'] = round((df['y_pred_prob'] - y_test) / y_test, 3) * 100
                df_results = df_results.append(df, ignore_index=True)
                with open('stats_products.textmate', 'w') as file:
                    file.write(str(df_statistics) + '\n')
                #print(df.describe())
        df_results = df_results.drop_duplicates(subset=['offer-ID'], keep='last')
        with open('all_results.textmate', 'w') as file:
            file.write(str(df_results) + '\n')
        del df_results['MLA']

    return df_results


main()