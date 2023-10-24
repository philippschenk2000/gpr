# math & other stuff
import datetime as datetime
import random
from random import choice
from random import randint
import scipy.stats as stats

# formalization & visualization
import seaborn as sns
import warnings
import re
from time import sleep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Web scraping using Scrapy
import requests
from scrapy import Selector

# machine learning
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model


def main():
    warnings.filterwarnings("ignore")
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    urls = []
    agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.104 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:84.0) Gecko/20100101 Firefox/84.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.2 Safari/605.1.15',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36',
        'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:84.0) Gecko/20100101 Firefox/84.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64; rv:84.0) Gecko/20100101 Firefox/84.0',
        'Mozilla/4.0 (compatible; MSIE 9.0; Windows NT 6.1)',
        'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)',
        'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (Windows NT 6.2; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (Windows NT 6.2; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (Windows NT 6.2; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (Windows NT 6.2; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.0; Trident/5.0)',
        'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0',
        'Mozilla/5.0 (Windows NT 6.1; Win64; x64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)',
        'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)',
        'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; .NET CLR 2.0.50727; .NET CLR 3.0.4506.2152; .NET CLR 3.5.30729)',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'AppleWebKit/537.36 (KHTML, like Gecko)',
        'Chrome/63.0.3239.132 Safari/537.36',
        'Mozilla/5.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.7 (KHTML, like Gecko) Chrome/16.0.912.36 Safari/535.7',
        'Mozilla/5.0 (Windows NT 6.2; Win64; x64; rv:16.0) Gecko/16.0 Firefox/16.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/534.55.3 (KHTML, like Gecko) Version/5.1.3 Safari/534.53.10']

    # FIRST PHASE - USER DATA
    start_page = 1
    end_page = 2
    for i in range(0, 1):

        # Creating search urls
        for i in range(start_page, end_page+1):
            urls.append("https://www.vinexus.de/wein/?p={}&n=120".format(i))

        # Finding the offer urls
        for i in urls:
            user_agent = choice(agents)
            headers = {'User-Agent': user_agent}
            html = requests.get(url=i, headers=headers).content
            sel = Selector(text=html)
            getofferurls = getofferurl(sel)

            print('Die Anzeigen und deren dazugehörigen User werden sich von folgender Seite angeschaut...')
            print(i)
            # Finding for each offer-url the user-url
            for y in getofferurls:
                #y = 'https://www.vinexus.de/chateau-de-saint-cosme-little-james-basket-press-rouge.html?c=2956328'
                # Scraping for each user-url with an offer from the search all profile data
                for f in range(0, 1):
                    i = y
                    i11 = [];  titel11 = [];  geschmack11 = [];  lieferant11 = [];  land11 = [];  region11 = [];  kategorie11 = [];  beschreibung11 = [];  bild11 = [];  preis11 = [];  preis_alt11 = [];  preis_pro_liter11 = [];  lieferzeit11 = [];  favorit11 = []; beschreibung21 = [];  beschreibung31 = [];  beschreibung41 = [];  expertise11 = [];  titel21 = [];  land21 = [];  region21 = [];  jahrgang11 = [];  rebsorte11 = [];  rebsorbenanteil11 = [];  kellermeister11 = [];  qualitaetsbezeichnung11 = [];  klassifikation11 = [];  alkoholgehalt11 = [];  saeuregehalt11 = [];  restsuesse11 = [];  verschlusstyp11 = [];  groesse11 = [];  zum_essen11 = [];  gebindegroesse11 = [];  empfehlung11 = [];  weingut11 = [];  weingut_beschreibung11 = [];  auszeichnungen11 = [];  auszeichnung11 = [];  auszeichnung21 = [];  auszeichnung31 = [];  auszeichnung41 = [];  auszeichnung51 = [];  auszeichnung01 = [];  empfehlung1_name11 = [];  empfehlung1_datum11 = [];  empfehlung1_titel11 = [];  empfehlung1_inhalt11 = [];  empfehlung2_name11 = [];  empfehlung2_datum11 = [];  empfehlung2_titel11 = [];  empfehlung2_inhalt11 = [];  inhalt11 = [];  inhalt21 = [];  inhalt31 = [];  inhalt41 = []
                    user_agent = choice(agents)
                    headers = {'User-Agent': user_agent}
                    html = requests.get(url=i, headers=headers).content
                    sel = Selector(text=html)
                    # now we want to get all data scraped for this one user / data filtering
                    profildata = getprofiles(sel, i, i11, titel11, geschmack11, lieferant11, land11, region11, kategorie11, beschreibung11, bild11, preis11, preis_alt11, preis_pro_liter11, lieferzeit11, favorit11, beschreibung21, beschreibung31, beschreibung41, expertise11, titel21, land21, region21, jahrgang11, rebsorte11, rebsorbenanteil11, kellermeister11, qualitaetsbezeichnung11, klassifikation11, alkoholgehalt11, saeuregehalt11, restsuesse11, verschlusstyp11, groesse11, zum_essen11, gebindegroesse11, empfehlung11, weingut11, weingut_beschreibung11, auszeichnungen11, auszeichnung11, auszeichnung21, auszeichnung31, auszeichnung41, auszeichnung51, auszeichnung01, empfehlung1_name11, empfehlung1_datum11, empfehlung1_titel11, empfehlung1_inhalt11, empfehlung2_name11, empfehlung2_datum11, empfehlung2_titel11, empfehlung2_inhalt11, inhalt11, inhalt21, inhalt31, inhalt41)

                    for i in range(0, 1):
                        df_profiles = pd.read_csv('csv_data/profiles.csv')
                        with open('1:df_profiles.textmate', 'w') as file:
                            file.write(str(df_profiles) + '\n')
                        filled_data = df_profiles.drop_duplicates(keep='first')
                        with open('1:filled_df_profiles.textmate', 'w') as file:
                            file.write(str(filled_data) + '\n')
                        filled_data.to_csv('csv_data/profiles_old.csv', index=False)

                sleeptime = float(randint(12, 16))
                sleep(sleeptime)


def getofferurl(sel):

    links = []
    links = sel.css('a.product--title::attr(href)').extract()

    return links


def getprofiles(sel, i, i11, titel11, geschmack11, lieferant11, land11, region11, kategorie11, beschreibung11, bild11, preis11, preis_alt11, preis_pro_liter11, lieferzeit11, favorit11, beschreibung21, beschreibung31, beschreibung41, expertise11, titel21, land21, region21, jahrgang11, rebsorte11, rebsorbenanteil11, kellermeister11, qualitaetsbezeichnung11, klassifikation11, alkoholgehalt11, saeuregehalt11, restsuesse11, verschlusstyp11, groesse11, zum_essen11, gebindegroesse11, empfehlung11, weingut11, weingut_beschreibung11, auszeichnungen11, auszeichnung11, auszeichnung21, auszeichnung31, auszeichnung41, auszeichnung51, auszeichnung01, empfehlung1_name11, empfehlung1_datum11, empfehlung1_titel11, empfehlung1_inhalt11, empfehlung2_name11, empfehlung2_datum11, empfehlung2_titel11, empfehlung2_inhalt11, inhalt11, inhalt21, inhalt31, inhalt41):

    df = pd.DataFrame()
    df1 = pd.DataFrame()
    for j in range(0, 1):
        if len(sel.css('.product--title::text').extract()) < 1:
            print('Vinexus blockt noch diese Anfrage oder user wurde gelöscht:')
            print(i)
            name = ''
            break
        else:
            print(i)
            titel = sel.css('.product--title::text').extract_first()
            #print(titel)
            geschmack = sel.css('.property-taste::text').extract_first()
            if geschmack != None:
                geschmack = re.sub(r'[!@#$(),/"%^*?:.;~`0-9 ]', '', geschmack)  # removing the symbols and numbers
            #print(geschmack)
            lieferant = sel.css('.property-supplier::text').extract_first()
            lieferant = re.sub(r'[!@#$(),/"%^*?:.;~`0-9]', '', lieferant)  # removing the symbols and numbers
            #print(lieferant)
            land = sel.css('.property-country::text').extract_first()
            land = re.sub(r'[!@#$(),/"%^*?:.;~`0-9 ]', '', land)  # removing the symbols and numbers
            #print(land)
            region = sel.css('.property-region::text').extract_first()
            if region != None:
                region = re.sub(r'[!@#$(),/"%^*?:.;~`0-9 ]', '', region)  # removing the symbols and numbers
            #print(region)
            kategorie = sel.css('.property-category::text').extract_first()
            if kategorie != None:
                kategorie = re.sub(r'[!@#$(),/"%^*?:.;~`0-9 ]', '', kategorie)  # removing the symbols and numbers
            #print(kategorie)
            beschreibung = sel.css('.product--detail-description-short > p:nth-child(2)::text').extract_first()
            #print(beschreibung)
            bild = sel.css('.image--media > img:nth-child(1)::attr(srcset)').extract()
            if bild != None:
                bild = bild[0]
            #print(bild)

            price1 = sel.css('.price--content::text').extract()
            preis00 = []
            for sub in price1:
                if sub != '':
                    preis00.append(sub.replace("\xa0", ""))
            preis0 = []
            for sub in preis00:
                if sub != '':
                    preis0.append(sub.replace(" ", ""))
            preis = []
            for x in preis0:
                if x != '':
                    preis.append(re.sub(r'[!@#$()/"%^*?:.;~` ]', '', x))
            preis = preis[0]
            #print(preis)

            preis_old1 = sel.css('.price--line-through::text').extract_first()
            #print(preis_old1)
            preis00 = []
            if preis_old1 != None:
                preis00.append(preis_old1.replace("\xa0", ""))
            else:
                preis00.append(None)
            preis_alt = []
            for x in preis00:
                if x != '':
                    if x != None:
                        preis_alt.append(re.sub(r'[!@#$()/"%^*?:.;~` ]', '', x))
                        preis_alt = preis_alt[0]
                else:
                    preis_alt = None
            #print(preis_alt)

            price_per_liter = sel.css('div.product--price:nth-child(1)::text').extract()
            preis00 = []
            for sub in price_per_liter:
                if sub != '':
                    preis00.append(sub.replace("\xa0", ""))
            preis0 = []
            for sub in preis00:
                if sub != ' ':
                    preis0.append(sub.replace("", ""))
            preis_pro_liter = []
            for x in preis0:
                if x != '':
                    preis_pro_liter.append(re.sub(r'[!@#$"%^*?:;~` ]', '', x))
            preis_pro_liter = preis_pro_liter[0]
            #print(preis_pro_liter)

            delivery = sel.css('.stockText::text').extract()
            lieferzeit = []
            for sub in delivery:
                if sub != '':
                    lieferzeit.append(sub.replace("", ""))
            lieferzeit = lieferzeit[1]
            #print(lieferzeit)


            favorit = sel.css('.hg--tag::text').extract()
            if len(favorit)> 0:
                favorit = favorit[1]
            #print(favorit)
            beschreibung2 = sel.css('.description-box-description > p:nth-child(2)::text').extract_first()
            #print(beschreibung2)
            beschreibung3 = sel.css('.description-box-description > p:nth-child(3)::text').extract_first()
            #print(beschreibung3)
            beschreibung4 = sel.css('.description-box-description > p:nth-child(4)::text').extract_first()
            #print(beschreibung4)
            beschreibung5 = sel.css('.description-box-description > p:nth-child(5)::text').extract_first()
            #print(beschreibung5)

            expertise = sel.css('#sp-expertise > div:nth-child(1) > h3:nth-child(2)::text').extract_first()
            #print(expertise)
            titel2 = sel.css('div.expertise-box:nth-child(1) > dl:nth-child(1) > dd:nth-child(2) > a:nth-child(1)::text').extract_first()
            #print(titel2)
            land2 = sel.css('div.expertise-box:nth-child(1) > dl:nth-child(1) > dd:nth-child(4) > a:nth-child(1)::text').extract_first()
            #print(land2)
            region2 = sel.css('div.expertise-box:nth-child(1) > dl:nth-child(1) > dd:nth-child(6) > a:nth-child(1)::text').extract_first()
            #print(region2)
            jahrgang = sel.css('div.expertise-box:nth-child(1) > dl:nth-child(1) > dd:nth-child(8)::text').extract_first()
            #print(jahrgang)
            rebsorte = sel.css('div.expertise-box:nth-child(1) > dl:nth-child(1) > dd:nth-child(10)::text').extract_first()
            #print(rebsorte)
            rebsorbenanteil = sel.css('div.expertise-box:nth-child(1) > dl:nth-child(1) > dd:nth-child(12)::text').extract_first()
            #print(rebsorbenanteil)
            kellermeister = sel.css('div.expertise-box:nth-child(1) > dl:nth-child(1) > dd:nth-child(14)::text').extract_first()
            #print(kellermeister)
            qualitaetsbezeichnung = sel.css('div.expertise-box:nth-child(1) > dl:nth-child(1) > dd:nth-child(16)::text').extract_first()
            #print(qualitaetsbezeichnung)
            klassifikation = sel.css('div.expertise-box:nth-child(1) > dl:nth-child(1) > dd:nth-child(18)::text').extract_first()
            #print(klassifikation)

            alkoholgehalt = sel.css('div.expertise-box:nth-child(3) > dl:nth-child(1) > dd:nth-child(2)::text').extract_first()
            #print(alkoholgehalt)
            saeuregehalt = sel.css('div.expertise-box:nth-child(3) > dl:nth-child(1) > dd:nth-child(4)::text').extract_first()
            #print(saeuregehalt)
            restsuesse = sel.css('div.expertise-box:nth-child(3) > dl:nth-child(1) > dd:nth-child(6)::text').extract_first()
            #print(restsuesse)
            infos1 = sel.css('div.expertise-box:nth-child(1) > dl:nth-child(1)').extract()
            if len(infos1)>0:
                infos1 = infos1[0]
                #print(infos1)
                infos1 = infos1.replace('<dd>', '')
                infos1 = infos1.replace('<dl>', '')
                infos1 = infos1.replace('<dt>', '')
                infos1 = infos1.replace('</dd>', '')
                infos1 = infos1.replace('</dl>', '')
                infos1 = infos1.replace('</dt>', ':')
                infos1 = infos1.replace('<a', '')
                infos1 = infos1.replace('</a', '')
                infos1 = infos1.replace('href="', '')
                infos1 = infos1.replace('"', '')
                infos1 = infos1.replace('/', '')
                infos1 = infos1.replace('>', '')
                infos1 = infos1.replace('title=', '')
                infos1 = infos1.split("Jahr").pop()
                infos1 = infos1.replace('gang', 'Jahrgang')
                infos1 = str('Wein: '+str(titel)+' Hersteller :'+str(lieferant)+ ' Land: '+str(land)+ ' Region: '+str(region)+infos1)
                #print(infos1)

                infos2 = sel.css('div.expertise-box:nth-child(3) > dl:nth-child(1)').extract()
                infos2 = infos2[0]
                infos2 = infos2.replace('<dd>','')
                infos2 = infos2.replace('<dl>','')
                infos2 = infos2.replace('<dt>','')
                infos2 = infos2.replace('</dd>','')
                infos2 = infos2.replace('</dl>','')
                infos2 = infos2.replace('</dt>',':')
                #print(infos2)
                infos = str(infos1+infos2)
                #print(infos)
            else:
                infos=[]
            groesse = sel.css('div.expertise-box:nth-child(3) > dl:nth-child(1) > dd:nth-child(10)::text').extract_first()
            #print(groesse)
            zum_essen = sel.css('div.expertise-box:nth-child(3) > dl:nth-child(1) > dd:nth-child(12)::text').extract_first()
            #print(zum_essen)
            gebindegroesse = sel.css('div.expertise-box:nth-child(3) > dl:nth-child(1) > dd:nth-child(14)::text').extract_first()
            #print(gebindegroesse)
            empfehlung = sel.css('.tb-expertise-subtext::text').extract_first()
            #print(empfehlung)

            weingut_bild = sel.css('div.tb-winery:nth-child(4) > div:nth-child(2) > img:nth-child(1)::attr(src)').extract()
            if weingut_bild != None:
                weingut_bild = weingut_bild[0]
            #print(weingut_bild)
            weingut = sel.css('#sp-winery > div:nth-child(1) > h3:nth-child(2)::text').extract_first()
            #print(weingut)
            weingut_beschreibung = sel.css('#sp-winery > div:nth-child(1) > p:nth-child(6)::text').extract_first()
            #print(weingut_beschreibung)

            auszeichnung0 = sel.css('.attr7::text').extract()
            #print(auszeichnung0)

            liste1 = []
            liste2 = []
            length = 0
            for x in range(1, len(auszeichnung0)+1):
                liste2 = []
                for a in range(1, 7):
                    y = 'ul.reward-liste:nth-child({}) > li:nth-child({})::text'.format(x, a)
                    auszeichnung = sel.css(y).extract_first()
                    if auszeichnung == None:
                        break
                    else:
                        liste2.append(auszeichnung)
                        length = length + 1
                liste1.append(liste2)
            #print(liste2)
            #print(length)


            empfehlung1_name = sel.css('div.review--entry:nth-child(2) > div:nth-child(1) > a:nth-child(4) > span:nth-child(1)::text').extract_first()
            #print(empfehlung1_name)
            empfehlung1_datum = sel.css('span.content--field:nth-child(7)::text').extract_first()
            #print(empfehlung1_datum)
            empfehlung1_titel = sel.css('h4.content--title::text').extract_first()
            #print(empfehlung1_titel)
            empfehlung1_inhalt = sel.css('.content--box::text').extract_first()
            #print(empfehlung1_inhalt)

            empfehlung2_name = sel.css('div.review--entry:nth-child(3) > div:nth-child(1) > span:nth-child(4)::text').extract_first()
            #print(empfehlung2_name)
            empfehlung2_datum = sel.css('div.review--entry:nth-child(3) > div:nth-child(1) > span:nth-child(7)::text').extract_first()
            #print(empfehlung2_datum)
            empfehlung2_titel = sel.css('div.review--entry:nth-child(3) > div:nth-child(2) > h4:nth-child(1)::text').extract_first()
            #print(empfehlung2_titel)
            empfehlung2_inhalt = sel.css('div.review--entry:nth-child(3) > div:nth-child(2) > p:nth-child(2)::text').extract_first()
            #print(empfehlung2_inhalt)

            empfehlung3_name = sel.css('div.review--entry:nth-child(4) > div:nth-child(1) > a:nth-child(4) > span:nth-child(1)::text').extract_first()
            #print(empfehlung3_name)
            empfehlung3_datum = sel.css('div.review--entry:nth-child(4) > div:nth-child(1) > span:nth-child(7)::text').extract_first()
            #print(empfehlung3_datum)
            empfehlung3_titel = sel.css('div.review--entry:nth-child(4) > div:nth-child(2) > h4:nth-child(1)::text').extract_first()
            #print(empfehlung3_titel)
            empfehlung3_inhalt = sel.css('div.review--entry:nth-child(4) > div:nth-child(2) > p:nth-child(2)::text').extract_first()
            #print(empfehlung3_inhalt)

            empfehlung4_name = sel.css('div.review--entry:nth-child(5) > div:nth-child(1) > span:nth-child(4)::text').extract_first()
            #print(empfehlung4_name)
            empfehlung4_datum = sel.css('div.review--entry:nth-child(5) > div:nth-child(1) > span:nth-child(7)::text').extract_first()
            #print(empfehlung4_datum)
            empfehlung4_titel = sel.css('div.review--entry:nth-child(5) > div:nth-child(2) > h4:nth-child(1)::text').extract_first()
            #print(empfehlung4_titel)
            empfehlung4_inhalt = sel.css('div.review--entry:nth-child(5) > div:nth-child(2) > p:nth-child(2)::text').extract_first()
            #print(empfehlung4_inhalt)

            inhalte = sel.css('div.maxcontainer::text').extract()
            inhalt0 = inhalte[-5]
            #print(inhalt0)
            inhalt1 = inhalte[-4]
            #print(inhalt1)
            inhalt2 = inhalte[-3]
            #print(inhalt2)
            inhalt3 = inhalte[-2]
            #print(inhalt3)
            inhalt4 = inhalte[-1]
            #print(inhalt4)

            '''
            'titel2':titel2,
            'land2':land2,
            'region2':region2,
            'jahrgang':jahrgang,
            'rebsorte':rebsorte,
            'rebsorbenanteil':rebsorbenanteil,
            'kellermeister':kellermeister,
            'qualitaetsbezeichnung':qualitaetsbezeichnung,
            'klassifikation':klassifikation,
            'alkoholgehalt':alkoholgehalt,
            'saeuregehalt':saeuregehalt,
            'restsuesse':restsuesse,
            'verschlusstyp':verschlusstyp,
            'groesse':groesse,
            'zum_essem': zum_essen,
            'gebindegroesse':gebindegroesse,
            '''

            df1 = df.append({
                'i': i,
                'titel': titel,
                'geschmack':geschmack,
                'lieferant':lieferant,
                'land':land,
                'region':region,
                'kategorie':kategorie,
                'beschreibung':beschreibung,
                'bild':bild,
                'preis':preis,
                'preis_alt':preis_alt,
                'preis_pro_liter':preis_pro_liter,
                'lieferzeit':lieferzeit,
                'favorit':favorit,
                'beschreibung2':beschreibung2,
                'beschreibung3':beschreibung3,
                'beschreibung4':beschreibung4,
                'beschreibung5': beschreibung5,
                'expertise':expertise,
                'infos':infos,
                'empfehlung': empfehlung,
                'weingut_bild': weingut_bild,
                'weingut': weingut,
                'weingut_beschreibung': weingut_beschreibung,
                'auszeichnungen': length,
                'auszeichnung_titel': auszeichnung0,
                'auszeichnung_inhalt': liste1,
                'empfehlung1_name': empfehlung1_name,
                'empfehlung1_datum': empfehlung1_datum,
                'empfehlung1_titel': empfehlung1_titel,
                'empfehlung1_inhalt': empfehlung1_inhalt,
                'empfehlung2_name': empfehlung2_name,
                'empfehlung2_datum': empfehlung2_datum,
                'empfehlung2_titel': empfehlung2_titel,
                'empfehlung2_inhalt': empfehlung2_inhalt,
                'empfehlung3_name': empfehlung3_name,
                'empfehlung3_datum': empfehlung3_datum,
                'empfehlung3_titel': empfehlung3_titel,
                'empfehlung3_inhalt': empfehlung3_inhalt,
                'empfehlung4_name': empfehlung4_name,
                'empfehlung4_datum': empfehlung4_datum,
                'empfehlung4_titel': empfehlung4_titel,
                'empfehlung4_inhalt': empfehlung4_inhalt,
                'inhalt0': inhalt0,
                'inhalt1': inhalt1,
                'inhalt2': inhalt2,
                'inhalt3': inhalt3,
                'inhalt4': inhalt4,

            }, ignore_index=True)
            df1 = df1.replace('[]', None).astype(str)

            with open('1:df', 'w') as file:
                file.write(str(df1) + '\n')
            df = pd.read_csv('csv_data/profiles.csv')
            df = df.append(df1)
            df.to_csv('csv_data/profiles.csv', index=False)

    return df1


def analysis_correl(filled_data):

    corr = filled_data.corr(method='pearson')
    plt.figure(figsize=(20, 6))
    sns.heatmap(corr, annot=True, cmap='Blues')
    plt.title('Correlation matrix')
    plt.show()

    return corr


main()