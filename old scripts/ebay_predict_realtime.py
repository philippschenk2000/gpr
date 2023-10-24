import datetime as datetime
import re
import warnings
from random import choice
from random import randint
from time import sleep
import numpy as np
import pandas as pd
import requests
import spacy
import spacy.cli
import spacy.lang.de
from requests.exceptions import ChunkedEncodingError
from scrapy import Selector
from simpledbf import Dbf5
from sklearn import ensemble
import tkinter as tk
from tkinter import ttk
#spacy.cli.download("de_core_news_sm")
#init_notebook_mode(connected=True)
#plotly.offline.init_notebook_mode(connected=True)


def main():

    warnings.filterwarnings("ignore")
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
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

    #offer_id = 2333099825 #int(input('Anzeigen-ID: '))

    root = tk.Tk()
    root.geometry('500x300')
    root.resizable(True, True)
    root.title('Dein Betrugsrechner für ebay-kleinanzeigen.de')

    def login_clicked():
        global plz
        global plzz
        global land
        global land1
        offer_idd = plz.get()
        print(offer_idd)
        x = main2(offer_idd, agents)
        print(x)
        x['y_pred_prob'] = x['y_pred_prob']*100
        x['y_pred_prob'] = np.around(x['y_pred_prob'], 2)
        x['y_pred_prob'] = x['y_pred_prob'].astype(str)
        x = x['y_pred_prob'].astype(str)
        x = x.reset_index()
        x = x['y_pred_prob'].astype(str)
        x = x[0]
        result1.delete(0, 200)
        result1.insert(1, str(x))
    global land
    global land1
    global plz
    global offer_idd
    plzs = ttk.Label(text="Geben Sie die Anzeigen-ID hier ein: ", font=("Helvetica", 14))
    plzs.pack(pady=(10, 15))
    plz = tk.StringVar()
    plz = ttk.Entry(textvariable=plz, background='white', foreground='white', width=10)
    plz.pack(pady=(0, 20))
    offer_idd = plz.get()

    login_button = ttk.Button(text="Betrugswahrscheinlichkeit berechnen", command=login_clicked)
    login_button.pack(pady=(20, 15))
    login_button.focus()

    result = ttk.Label(text="Betrugswahrscheinlichkeit in %", font=("Helvetica", 14))
    result.pack(pady=(20, 15))
    result1 = tk.StringVar()
    result1 = ttk.Entry(textvariable=result1, background='white', foreground='lime', width=6)
    result1.pack()
    root.mainloop()


def main2(offer_idd, agents):
    offer_id = int(offer_idd)

    getofferurls = []
    for i in range(0, 1):
    for i in getofferurls:
        user_agent = choice(agents)
        headers = {'User-Agent': user_agent}
        newest_offer = i
        html = requests.get(url=i, headers=headers).content
        sel = Selector(text=html)
        profiles = sel.css('html body#vap div.site-base div.site-base--content div#site-content.l-page-wrapper.l-container-row section#viewad-main.l-container-row section#viewad-cntnt.l-row.a-double-margin.l-container-row aside#viewad-sidebar.a-span-8.l-col div#viewad-profile-box.l-container-row.contentbox--vip.no-shadow.j-sidebar-content div#viewad-contact div.l-container-row ul.iconlist li span.iconlist-text span.text-body-regular-strong.text-force-linebreak a::attr(href)').extract()
        if len(profiles) > 0:

            for f in range(0, 1):
                i = y
                user_agent = choice(agents)
                headers = {'User-Agent': user_agent}
                trycnt = 5  # max try cnt
                while trycnt > 0:
                    try:
                        html = requests.get(url=i, headers=headers).content
                        sel = Selector(text=html)
                        df = getprofiles(sel, i)
                        df_offer = profileswithoffers(df, headers, newest_offer)
                        beautiful_df = editprofiledata(df_offer)
                        more_beautiful_df = filledprofiledata(beautiful_df)
                        df_profiles_offers = cleandata(more_beautiful_df, newest_offer)

                        apple_offers = filter_df_profiles_offers(df_profiles_offers)
                        apple_offers = latlong(apple_offers)  # Location information (Berlin Kreuzberg, Berlin,  52.5323,  13.3846)
                        mainproductclass = product_classifiing1(apple_offers)  # Product class (iphone)
                        detaillproductclass = product_classifiing2(mainproductclass)  # Exact product model (iphone 12 pro max)
                        mostdetaillproductclass = product_classifiing3(detaillproductclass)  # Capacity (128)

                        titles_analysed = titles_analysis(apple_offers)  # Written expressions (emojis, uppers,..)
                        descriptions_analysed = descriptions_analysis(apple_offers)             # Spelling & written expressions (verkauven, emojis,..)
                        location_analysed = location_analysis(apple_offers)  # Relationship with lat/lng/plz/bundesland and fraud
                        times_analysed = times_analysis(apple_offers)
                        profile_analysed = profiles_analysis(apple_offers)

                        ''' FINAL DATASET '''
                        for iii in range(0, 1):
                            # PRODUCT CLASSES & TITLES
                            final_data = pd.merge(mostdetaillproductclass, titles_analysed, on='offer-ID', how='left')
                            # PICTURES
                            # TIMING
                            final_data = pd.merge(final_data, times_analysed, on='offer-ID', how='left')
                            # PROFILES
                            final_data = pd.merge(final_data, profile_analysed, on='offer-ID', how='left')
                            # LOCATIONS
                            final_data = pd.merge(final_data, location_analysed, on='offer-ID', how='left')
                            # DESCRIPTIONS
                            final_data = pd.merge(final_data, descriptions_analysed, on='offer-ID', how='left')
                            final_data = final_data.fillna(0)


                            ''' Predicting '''
                            for iiii in range(0, 1):
                                data = pd.read_csv('csv_data/final_data.csv')
                                data = data.sample(frac=1).reset_index(drop=True)
                                for mmm in range(0, 1):
                                    del final_data['apple-offer-url'], final_data['pics'], final_data['titles'], \
                                    final_data['shippings_x'], final_data[
                                        'dates'], final_data['descriptions'], final_data['user-ID'], final_data['name'], \
                                    final_data[
                                        'startyear_y'], final_data['startweekday_y'], final_data['startmonth_y'], \
                                    final_data['startdayofmonth_y'], final_data[
                                        'profilerating_y'], final_data['prices_y'], final_data['shippings_y'], \
                                    final_data['Sicher_bezahlen_y'], final_data['offeringssum_y'], final_data[
                                        'profilefriendliness_y'], final_data['profilereliability_y'], final_data[
                                        'profilereplyrate_y'], final_data['profilereplyspeed_y'], \
                                        final_data['profilefollowers_y'], final_data['PLZ_y'], final_data['STADT_y'], \
                                    final_data['BUNDESLAND_y'], final_data['LAT_y'], \
                                        final_data['LNG_y'], final_data['prices_x'], \
                                        final_data['Gewerblicher_user_y'], final_data['offeringsonline_y'], final_data[
                                        'text_y'], \
                                        final_data['scrape_time'], final_data['apple_url'], \
                                    final_data['PLZ_x'], final_data['product'], \
                                    final_data['day'], final_data['month'], final_data['year'], final_data['text_x'], \
                                    final_data['NUTS3'], final_data['NUTS2']
                                    final_data.rename(columns={'Sicher_bezahlen_x': 'Sicher_bezahlen',
                                                               'Gewerblicher_user_x': 'Gewerblicher_user',
                                                               'offeringsonline_x': 'offeringsonline',
                                                               'offeringssum_x': 'offeringssum',
                                                               'profilefriendliness_x': 'profilefriendliness',
                                                               'profilereliability_x': 'profilereliability'}, inplace=True)
                                    final_data.rename(columns={'profilerating_x': 'profilerating',
                                                                'profilereplyrate_x': 'profilereplyrate',
                                                               'profilereplyspeed_x': 'profilereplyspeed',
                                                               'profilefollowers_x': 'profilefollowers',
                                                               'startyear_x': 'startyear'}, inplace=True)
                                    final_data.rename(columns={'startweekday_x': 'startweekday',
                                                               'startdayofmonth_x': 'startdayofmonth',
                                                               'startmonth_x': 'startmonth',
                                                               'STADT_x': 'STADT',
                                                               'LAT_x': 'LAT',
                                                               'LNG_x': 'LNG',
                                                               'BUNDESLAND_x': 'BUNDESLAND'}, inplace=True)
                                    final_data.rename(columns={'numb_share_x': 'title_numb_share',
                                                               'exclam_mark_share_x': 'title_exclam_mark_share',
                                                               'upper_share_x': 'title_upper_share',
                                                               'emojis_share_y': 'desc_emojis_share',
                                                               'emojis_y': 'desc_emojis',
                                                               'emojis_share_x': 'title_emojis_share',
                                                               'emojis_x': 'title_emojis',
                                                               'numb_share_y': 'desc_numb_share',
                                                               'exclam_mark_share_y': 'desc_exclam_mark_share',
                                                               'upper_share_y': 'desc_upper_share'
                                                               }, inplace=True)
                                    final_data.rename(columns={'characters': 'desc_characters', 'numbs': 'desc_numbs',
                                                         'exclam_marks': 'desc_exclam_marks', 'uppers': 'desc_uppers',
                                                         'words': 'desc_words', 'spacy_spelling': 'desc_spacy_spelling',
                                                         'paypal': 'desc_paypal',
                                                         'paypal_freunde': 'desc_paypal_freunde',
                                                         'mistake_rate': 'desc_mistake_rate'}, inplace=True)
                                    final_data.rename(columns={'length': 'title_length',
                                                               'numb_count': 'title_numb_count',
                                                               'exclam_mark_count': 'title_exclam_mark_count',
                                                               'upper_count': 'title_upper_count'}, inplace=True)
                                    del final_data['STADT'], final_data['BUNDESLAND'], final_data['einwohner'], final_data['capacity']
                                    del final_data['offer_day'], final_data['qkm'], final_data['desc_words']
                                    final_data = final_data.fillna(0)


                                    del data['pics'], data['titles'], data['dates'], data['descriptions'], data[
                                        'fraud_x'], data['scraptime'], data['STADT_x'], data['BUNDESLAND_x'], data[
                                        'product'], data['product_model'], data['text_x'], data['STADT_y'], data[
                                        'BUNDESLAND_y'], data['NUTS3'], data['NUTS2'], data['text_y'], data[
                                        'fraud_x.1'], data['fraud_y.1'], data['fraud_y'], data['prices_y'], data[
                                        'west'], data['qkm'], data['PLZ'], data['capacity'], data['einwohner'], data[
                                        'offer_day']

                                    data.rename(columns={'numb_share_x': 'title_numb_share',
                                                         'exclam_mark_share_x': 'title_exclam_mark_share',
                                                         'upper_share_x': 'title_upper_share',
                                                         'emojis_share_x': 'title_emojis_share',
                                                         'emojis_x': 'title_emojis', 'prices_x': 'prices'}, inplace=True)
                                    data.rename(columns={'length': 'title_length', 'numb_count': 'title_numb_count',
                                                         'exclam_mark_count': 'title_exclam_mark_count',
                                                         'upper_count': 'title_upper_count'}, inplace=True)
                                    data.rename(columns={'numb_share_y': 'desc_numb_share',
                                                         'exclam_mark_share_y': 'desc_exclam_mark_share',
                                                         'upper_share_y': 'desc_upper_share',
                                                         'emojis_share_y': 'desc_emojis_share',
                                                         'emojis_y': 'desc_emojis'}, inplace=True)
                                    data.rename(columns={'characters': 'desc_characters', 'numbs': 'desc_numbs',
                                                         'exclam_marks': 'desc_exclam_marks', 'uppers': 'desc_uppers',
                                                         'words': 'desc_words', 'spacy_spelling': 'desc_spacy_spelling',
                                                         'paypal': 'desc_paypal',
                                                         'paypal_freunde': 'desc_paypal_freunde',
                                                         'mistake_rate': 'desc_mistake_rate'}, inplace=True)

                                    cols = list(data)
                                    cols[-1], cols[0] = cols[0], cols[-1]
                                    data = data.reindex(columns=cols)
                                    del data['desc_words']
                                # data = data.head(1000)

                                # 2 SPLITTING THE DATA & INPUTS
                                X = data.drop('fraud', axis=1)
                                y = data['fraud']
                                threshold = 0.29
                                X_train = X.head(len(X) - 1)
                                y_train = y.head(len(y) - 1)
                                X_test = X.tail(1)
                                for mmm in range(0, 1):
                                    print('-' * 10)
                                    print(round(y_train.value_counts()[1] / len(y_train) * 100, 2), '% Frauds in the trainset')
                                    offer_ID = X_test['offer-ID'].reset_index(drop=True)
                                    X_test = X_test.append(final_data, ignore_index=True)
                                    X_test = X_test.fillna(0)
                                    X_test = X_test.drop('offer-ID', axis=1)
                                    X_test = X_test.tail(1)
                                    X_train = X_train.drop('offer-ID', axis=1)


                                MLA = [ensemble.ExtraTreesClassifier(criterion='entropy', max_depth=10, n_estimators=300, random_state=0)]

                                for u in range(0, 1):
                                    print('-' * 10)
                                    df = pd.DataFrame()
                                    X_test = X_test.reset_index()
                                    del X_test['index']
                                    print('Es handelt sich hierbei um Folgende Anzeige bzw. Folgenden Nutzer:')

                                    for j in range(0, len(MLA)):
                                        mla_now = MLA[j]
                                        mla_now.fit(X_train.values, y_train.values)
                                        y_pred_prob = mla_now.predict_proba(X_test)[:, 1]

                                        df['y_pred_prob'] = y_pred_prob
                                        if df['y_pred_prob'][0] > threshold:
                                            df['y_predicted'] = 1
                                        else:
                                            df['y_predicted'] = 0
                                        df['offer-ID'] = offer_id
                                        df['MLA'] = str(mla_now)[:20]
                                        global gesamter_df
                                        gesamter_df = df

                                        return df



                    except ChunkedEncodingError as ex:
                        if trycnt <= 0:
                            print("Failed to retrieve: " + str(y) + "\n" + str(ex))  # done retrying
                        else:
                            trycnt -= 1  # retry
                        sleeptime = float(randint(1, 2))
                        sleep(sleeptime)


def getofferurl(sel):

    links = []
    link = sel.css('a.ellipsis::attr(href)').extract()
    for x in link:
        links.append(x)

    return links


def getprofiles(sel, i):

    userid = []; name1 = []; paymentdetails1 = []; offeringsonline1 = []; offeringssum1 = []; profilerating1 = []; profilefriendliness1 = []
    profilereliability1 = []; profilereplyrate1 = []; profilereplyspeed1 = []; profilefollowers1 = []; offernames1 = []; location1 = []
    description1 = []; prices1 = []; offerdates1 = []; shipping1 = []; start1 = []; usertype1 = []; offerlinks1 = []
    for j in range(0, 1):
        if len(sel.css('html body#pstrads div.site-base div.site-base--content div#site-content.l-page-wrapper.l-container-row div.l-splitpage-flex div.l-splitpage-navigation section.l-container-row.contentbox.surface.userprofile.j-followeduser header.a-single-margin.l-container-row span.userprofile-details::text').extract()) < 1:
            print('Ebay-kleinanzeigen blockt noch diese Anfrage oder user wurde gelöscht')
            print(i)
            name = ''
            break
        else:
            paymentdetails = sel.css('.user-profile-secure-payment::text').extract() # Sicher bezahlen eingerichtet
            paymentdetails = sel.css('html body#pstrads div.site-base div.site-base--content div#site-content.l-page-wrapper.l-container-row div.l-splitpage-flex div.l-splitpage-navigation section.l-container-row.contentbox.surface.userprofile.j-followeduser header.a-single-margin.l-container-row span.user-profile-secure-payment::text').extract() # Sicher bezahlen eingerichtet
            if len(paymentdetails) > 0:
                paymentdetails = 'Sicherbezahleneingerichtet'
                name = sel.css('html body#pstrads div.site-base div.site-base--content div#site-content.l-page-wrapper.l-container-row div.l-splitpage-flex div.l-splitpage-navigation section.l-container-row.contentbox.surface.userprofile.j-followeduser header.a-single-margin.l-container-row h2.userprofile--name::text').extract()  # 'Rainer'
                if len(name) > 0:
                    name = name[0]
                else:
                    name = ''
                usertype = sel.css('span.userprofile-details:nth-child(7)::text').extract()  # Privater Nutzer
                start = sel.css('span.userprofile-details:nth-child(9)::text').extract()  # Aktiv seit 1.9.2009
                offeringscount = sel.css('span.userprofile-details:nth-child(11)::text').extract()  # 5 Anzeigen online / 522 gesamt
            else:
                name = sel.css('html body#pstrads div.site-base div.site-base--content div#site-content.l-page-wrapper.l-container-row div.l-splitpage-flex div.l-splitpage-navigation section.l-container-row.contentbox.surface.userprofile.j-followeduser header.a-single-margin.l-container-row h2.userprofile--name::text').extract()  # 'Rainer'
                if len(name) > 0:
                    name = name[0]
                else:
                    name = ''
                paymentdetails = 'keinSicherbezahlen'
                usertype = sel.css('span.userprofile-details:nth-child(3)::text').extract()  # Privater Nutzer
                start = sel.css('span.userprofile-details:nth-child(5)::text').extract()  # Aktiv seit 1.9.2009
                offeringscount = sel.css('span.userprofile-details:nth-child(7)::text').extract()  # 5 Anzeigen online / 522 gesamt

            profilerating = sel.css('html body#pstrads div.site-base div.site-base--content div#site-content.l-page-wrapper.l-container-row div.l-splitpage-flex div.l-splitpage-navigation section.l-container-row.contentbox.surface.userprofile.j-followeduser div.followuseritem-main ul.badges-iconlist li.userbadges-public-profile.userbadges-profile-rating div.iconlist-text::text').extract() # TOP Zufriedenheit
            if len(profilerating) > 0:
                profilerating = profilerating[0]
            profilefriendliness = sel.css('html body#pstrads div.site-base div.site-base--content div#site-content.l-page-wrapper.l-container-row div.l-splitpage-flex div.l-splitpage-navigation section.l-container-row.contentbox.surface.userprofile.j-followeduser div.followuseritem-main ul.badges-iconlist li.userbadges-public-profile.userbadges-profile-friendliness div.iconlist-text::text').extract() # Besonders freundlich
            if len(profilefriendliness) > 0:
                profilefriendliness = profilefriendliness[0]
            profilereliability = sel.css('html body#pstrads div.site-base div.site-base--content div#site-content.l-page-wrapper.l-container-row div.l-splitpage-flex div.l-splitpage-navigation section.l-container-row.contentbox.surface.userprofile.j-followeduser div.followuseritem-main ul.badges-iconlist li.userbadges-public-profile.userbadges-profile-reliability div.iconlist-text::text').extract() # Besonders zuverlässig
            if len(profilereliability) > 0:
                profilereliability = profilereliability[0]
            profilereplyrate = sel.css('html body#pstrads div.site-base div.site-base--content div#site-content.l-page-wrapper.l-container-row div.l-splitpage-flex div.l-splitpage-navigation section.l-container-row.contentbox.surface.userprofile.j-followeduser div.followuseritem-main ul.badges-iconlist li.userbadges-public-profile.userbadges-profile-replyRate div.iconlist-text::text').extract() # xxx% Antwortrate
            if len(profilereplyrate) > 0:
                profilereplyrate = profilereplyrate[0]
            profilereplyspeed = sel.css('html body#pstrads div.site-base div.site-base--content div#site-content.l-page-wrapper.l-container-row div.l-splitpage-flex div.l-splitpage-navigation section.l-container-row.contentbox.surface.userprofile.j-followeduser div.followuseritem-main ul.badges-iconlist li.userbadges-public-profile.userbadges-profile-replySpeed div.iconlist-text::text').extract() # 1h Antwortzeit
            if len(profilereplyspeed) > 0:
                profilereplyspeed = profilereplyspeed[0]
            profilefollowers = sel.css('html body#pstrads div.site-base div.site-base--content div#site-content.l-page-wrapper.l-container-row div.l-splitpage-flex div.l-splitpage-navigation section.l-container-row.contentbox.surface.userprofile.j-followeduser div.followuseritem-main ul.badges-iconlist li.userbadges-public-profile.userbadges-profile-followers div.iconlist-text::text').extract() # 27 Follower
            if len(profilefollowers) > 0:
                profilefollowers = profilefollowers[0]
            offernames = sel.css('a.ellipsis::text').extract()  # ['Elektro-Heckenschere Top Craft 661', 'Spinning Bike Aerobike 800', 'Asics Onitsuka Tiger Sneaker - 43,5', 'LED Nachtlicht', 'Snatch Schulrucksack']
            location = sel.css('div.aditem-main--top--left::text').extract() # ['13158 Rosenthal', '16515 Oranienburg', '13158 ...']
            description = sel.css('p.aditem-main--middle--description::text').extract() # ['Verkaufe eine gebrauchte Elekto-Heckenschere von der Firma Top Craft. Sie ist ca. 60cm lang und hat...', 'xxxx', ]
            prices = sel.css('p.aditem-main--middle--price-shipping--price::text').extract() # ['12 € VB', 'xy €', ]
            offerdates = sel.css('div.aditem-main--top--right::text').extract() # ['17.10.2022', 'xx.xx.xxxx', ]
            shipping = sel.css('p.aditem-main--middle--price-shipping--shipping::text').extract() # ['', 'Versand möglich', ]
            offerlinks = sel.css('a.ellipsis::attr(href)').extract()


        usertype3 = []
        for sub in usertype:
            usertype3.append(sub.replace("\n", ""))
        usertype0 = []
        for sub in usertype3:
            usertype0.append(sub.replace(" ", ""))

        start3 = []
        for sub in start:
            start3.append(sub.replace("\n", ""))
        start2 = []
        for sub in start3:
            start2.append(sub.replace(" ", ""))
        start0 = []
        for sub in start2:
            start0.append(sub.replace("Aktivseit", ""))
        start0 = start0[0]

        for i in range(0,1):
            if len(start0) == 0:
                break
            else:
                usertype0 = usertype0[0]

            offeringscount3 = []
            for sub in offeringscount:
                offeringscount3.append(sub.replace("\n", ""))
            offeringscount0 = []
            for sub in offeringscount3:
                offeringscount0.append(sub.replace(" ", ""))
            offeringscount0 = offeringscount0[0]
            x = re.split('[/]', offeringscount0)
            offeringsonline = x[0]
            offeringsonline = offeringsonline.replace('Anzeigenonline', '')
            if len(x) > 1:
                offeringssum = x[1]
                offeringssum = offeringssum.replace('gesamt', '')
            else:
                offeringssum = 0

            if len(profilefriendliness) > 0:
                profilefriendliness = profilefriendliness.replace(' ', '')
            if len(profilereliability) > 0:
                profilereliability = profilereliability.replace(' ', '')

            location3 = []
            for sub in location:
                location3.append(sub.replace("\n", ""))
            location2 = []
            for sub in location3:
                if sub != '':
                    location2.append(sub.replace(" ", ""))
            location0 = []
            for sub in location2:
                if sub != '':
                    location0.append(sub)

            description0 = []
            for sub in description:
                description0.append(sub.replace("\n", " "))

            prices3 = []
            for sub in prices:
                prices3.append(sub.replace("\n", ""))
            prices2 = []
            for sub in prices3:
                if sub != '':
                    prices2.append(sub.replace(" ", ""))
            prices0 = []
            for sub in prices2:
                if sub != '':
                    prices0.append(sub)

            offerdates3 = []
            for sub in offerdates:
                offerdates3.append(sub.replace("\n", ""))
            offerdates2 = []
            for sub in offerdates3:
                if sub != '':
                    offerdates2.append(sub.replace(" ", ""))
            offerdates0 = []
            for sub in offerdates2:
                if sub != '':
                    offerdates0.append(sub)

            shipping3 = []
            for sub in shipping:
                shipping3.append(sub.replace("\n", ""))
            shipping0 = []
            for sub in shipping3:
                if len(shipping3) < 1:
                    shipping0 = 'NurAbholung'
                else:
                    shipping0.append(sub.replace(" ", ""))

            name1.append(name)
            paymentdetails1.append(paymentdetails)
            usertype1.append(usertype0)
            start1.append(start0)
            offeringsonline1.append(offeringsonline)
            offeringssum1.append(offeringssum)
            profilerating1.append(profilerating)
            profilefriendliness1.append(profilefriendliness)
            profilereliability1.append(profilereliability)
            profilereplyrate1.append(profilereplyrate)
            profilereplyspeed1.append(profilereplyspeed)
            profilefollowers1.append(profilefollowers)
            offernames1.append(offernames)
            location1.append(location0)
            description1.append(description0)
            prices1.append(prices0)
            offerdates1.append(offerdates0)
            shipping1.append(shipping0)
            offerlinks1.append(offerlinks)

    df = pd.DataFrame((zip(userid, name1, paymentdetails1, usertype1, start1, offeringsonline1, offeringssum1, profilerating1, profilefriendliness1, profilereliability1, profilereplyrate1, profilereplyspeed1, profilefollowers1, offernames1, location1, description1, prices1, offerdates1, shipping1, offerlinks1)), columns=['user-ID', 'name', 'paymentdetails', 'usertype', 'start', 'offeringsonline', 'offeringssum', 'profilerating', 'profilefriendliness', 'profilereliability', 'profilereplyrate', 'profilereplyspeed', 'profilefollowers', 'offernames', 'location', 'description', 'prices', 'offerdates', 'shipping', 'offerlinks'])
    df['year'] = pd.DatetimeIndex(df['start']).year
    df['weekday'] = pd.DatetimeIndex(df['start']).weekday
    df['month'] = pd.DatetimeIndex(df['start']).day
    df['dayofmonth'] = pd.DatetimeIndex(df['start']).month
    df['scraptime'] = datetime.datetime.now()

    return df


def profileswithoffers(df, headers, newest_offer):

    df_offer = pd.DataFrame()
    if df['offerlinks'][0] != None:
        offerIDs = []
        pics = []
        titles = []
        prices = []
        shippings = []
        locations = []
        dates = []
        views = []
        descriptions = []
        for j in df['offerlinks']:
            for i in j:
                i = i.split('-')[-3]
                i = i.split('/')
                i = i[-1]
                html = requests.get(url=y, headers=headers).content
                sel = Selector(text=html)

                offerID = sel.css('.text-light-800 > li:nth-child(2)::text').extract()
                offerIDs.append(offerID)
                pic = sel.css('#viewad-image::attr(src)').extract()
                pics.append(pic)
                title = sel.css('#viewad-title::text').extract()
                if title == []:
                    title = ' '
                for sub in title:
                    sub = sub.replace("\n", " ")
                    titles.append(sub)
                price = sel.css('#viewad-price::text').extract()
                if price == []:
                    price = '0'
                for sub in price:
                    sub = sub.replace("\n", " ")
                    prices.append(sub)
                shipping = sel.css('.boxedarticle--details--shipping::text').extract()
                shippings.append(shipping)
                location = sel.css('#viewad-locality::text').extract()
                if location == []:
                    location = ['99999']
                location_new = []
                for sub in location:
                    sub = sub.replace("\n", " ")
                    sub = sub.replace("\xa0", "")
                    location_new.append(sub)
                for sub in location_new:
                    sub = sub.replace(" ", "")
                    locations.append(sub)
                date = sel.css('#viewad-extra-info > div:nth-child(1) > span:nth-child(2)::text').extract()
                dates.append(date)

                view = sel.css('#viewad-extra-info > div:nth-child(2)')#.extract() #viewad-cntr-num
                views.append(view)

                description = sel.css('#viewad-description-text::text').extract()
                xy = []
                for sub in description:
                    sub = sub.replace("\n", " ")
                    xy.append(sub)
                descriptions.append(xy)

        df_offer['offer-ID'] = offerIDs
        df_offer['apple-offer-url'] = newest_offer
        df_offer['pics'] = pics
        df_offer['titles'] = titles
        df_offer['prices'] = prices
        df_offer['shippings'] = shippings
        df_offer['locations'] = locations
        df_offer['locations'].fillna(method='ffill', inplace=True)
        df_offer['locations'].fillna(method='bfill', inplace=True)
        df_offer['dates'] = dates
        df_offer['views'] = views
        df_offer['descriptions'] = descriptions
        df_offer['user-ID'] = df['user-ID'][0]
        df_offer['name'] = df['name'][0]
        df_offer['paymentdetails'] = df['paymentdetails'][0]
        df_offer['usertype'] = df['usertype'][0]
        df_offer['start'] = df['start'][0]
        df_offer['offeringsonline'] = df['offeringsonline'][0]
        df_offer['offeringssum'] = df['offeringssum'][0]


        profilerating = []
        for i in range(0, len(df['profilerating'])):
            if df['profilerating'][i] != '[]':
                for j in range(0, len(df_offer['offer-ID'])):
                    profilerating.append(df['profilerating'][i])
            else:
                for j in range(0, len(df_offer['offer-ID'])):
                    profilerating.append(0)
        df_offer['profilerating'] = profilerating

        profilefriendliness = []
        for i in range(0, len(df['profilefriendliness'])):
            if df['profilefriendliness'][i] != '[]':
                for j in range(0, len(df_offer['offer-ID'])):
                    profilefriendliness.append(df['profilefriendliness'][i])
            else:
                for j in range(0, len(df_offer['offer-ID'])):
                    profilefriendliness.append(0)
        df_offer['profilefriendliness'] = profilefriendliness

        profilereliability = []
        for i in range(0, len(df['profilereliability'])):
            if df['profilereliability'][i] != '[]':
                for j in range(0, len(df_offer['offer-ID'])):
                    profilereliability.append(df['profilereliability'][i])
            else:
                for j in range(0, len(df_offer['offer-ID'])):
                    profilereliability.append(0)
        df_offer['profilereliability'] = profilereliability

        profilereplyrate = []
        for i in range(0, len(df['profilereplyrate'])):
            if df['profilereplyrate'][i] != '[]':
                for j in range(0, len(df_offer['offer-ID'])):
                    profilereplyrate.append(df['profilereplyrate'][i])
            else:
                for j in range(0, len(df_offer['offer-ID'])):
                    profilereplyrate.append(0)
        df_offer['profilereplyrate'] = profilereplyrate

        profilereplyspeed = []
        for i in range(0, len(df['profilereplyspeed'])):
            if df['profilereplyspeed'][i] != '[]':
                for j in range(0, len(df_offer['offer-ID'])):
                    profilereplyspeed.append(df['profilereplyspeed'][i])
            else:
                for j in range(0, len(df_offer['offer-ID'])):
                    profilereplyspeed.append(0)
        df_offer['profilereplyspeed'] = profilereplyspeed

        profilefollowers = []
        for i in range(0, len(df['profilefollowers'])):
            if df['profilefollowers'][i] != '[]':
                for j in range(0, len(df_offer['offer-ID'])):
                    profilefollowers.append(df['profilefollowers'][i])
            else:
                for j in range(0, len(df_offer['offer-ID'])):
                    profilefollowers.append(0)
        df_offer['profilefollowers'] = profilefollowers

    return df_offer


def editprofiledata(df_offer):

    # makes it more computional
    df_offer['year'] = pd.DatetimeIndex(df_offer['start']).year
    df_offer['weekday'] = pd.DatetimeIndex(df_offer['start']).weekday
    df_offer['month'] = pd.DatetimeIndex(df_offer['start']).day
    df_offer['dayofmonth'] = pd.DatetimeIndex(df_offer['start']).month
    df_offer['scraptime'] = datetime.datetime.now()

    df_offer['name'] = df_offer['name'].replace('[]', None).astype(str)
    name = []
    for x in df_offer['name']:
        if x == 'nan':
            x = 0
        else:
            x = 1
        name.append(x)
    df_offer['name'] = name

    df_offer['paymentdetails'] = df_offer['paymentdetails'].replace('Sicherbezahleneingerichtet', 1)
    df_offer['paymentdetails'] = df_offer['paymentdetails'].replace('keinSicherbezahlen', 0)
    df_offer['usertype'] = df_offer['usertype'].replace('GewerblicherNutzer', 1)
    df_offer['usertype'] = df_offer['usertype'].replace('PrivaterNutzer', 0)
    df_offer['usertype'] = df_offer['usertype'].replace('', 0)
    df_offer['offeringsonline'] = df_offer['offeringsonline'].astype(int)
    df_offer['offeringssum'] = df_offer['offeringssum'].astype(int)
    del df_offer['start']


    # build profilerating binary
    profilerating = []
    for i in range(0, len(df_offer['profilerating'])):
        if len(df_offer['profilerating'][i]) > 0:
            profilerating.append(df_offer['profilerating'][i])
        else:
            profilerating.append(None)
    df_offer['profilerating'] = profilerating
    df_offer['profilerating'] = df_offer['profilerating'].replace('TOP', 3)
    df_offer['profilerating'] = df_offer['profilerating'].replace('OK', 2)
    df_offer['profilerating'] = df_offer['profilerating'].replace('NA JA', 1)
    df_offer['profilerating'] = df_offer['profilerating'].astype(float)


    # build profilefriendliness binary
    profilefriendliness = []
    for i in range(0, len(df_offer['profilefriendliness'])):
        if len(df_offer['profilefriendliness'][i]) > 0:
            profilefriendliness.append(df_offer['profilefriendliness'][i])
        else:
            profilefriendliness.append(None)
    df_offer['profilefriendliness'] = profilefriendliness
    df_offer['profilefriendliness'] = df_offer['profilefriendliness'].replace('Besondersfreundlich', 3)
    df_offer['profilefriendliness'] = df_offer['profilefriendliness'].replace('Sehrfreundlich', 2)
    df_offer['profilefriendliness'] = df_offer['profilefriendliness'].replace('Freundlich', 1)
    df_offer['profilefriendliness'] = df_offer['profilefriendliness'].astype(float)


    # build profilereliability binary
    profilereliability = []
    for i in range(0, len(df_offer['profilereliability'])):
        if len(df_offer['profilereliability'][i]) > 0:
            profilereliability.append(df_offer['profilereliability'][i])
        else:
            profilereliability.append(None)
    df_offer['profilereliability'] = profilereliability
    df_offer['profilereliability'] = df_offer['profilereliability'].replace('Besonderszuverlässig', 3)
    df_offer['profilereliability'] = df_offer['profilereliability'].replace('Sehrzuverlässig', 2)
    df_offer['profilereliability'] = df_offer['profilereliability'].replace('Zuverlässig', 1)
    df_offer['profilereliability'] = df_offer['profilereliability'].astype(float)


    # build profilereplyrate binary
    profilereplyrate = []
    for i in df_offer['profilereplyrate']:
        if len(i) > 2:
            i = i.replace('%', '')
            profilereplyrate.append(i)
        else:
            profilereplyrate.append(None)
    df_offer['profilereplyrate'] = profilereplyrate
    df_offer['profilereplyrate'] = df_offer['profilereplyrate'].astype(float)


    # build profilereplyspeed binary
    profilereplyspeed = []
    for i in range(0, len(df_offer['profilereplyspeed'])):
        if len(df_offer['profilereplyspeed'][i]) > 0:
            df_offer['profilereplyspeed'][i] = df_offer['profilereplyspeed'][i].replace('h', '')
            df_offer['profilereplyspeed'][i] = df_offer['profilereplyspeed'][i].replace('10min', '0.16')
            profilereplyspeed.append(df_offer['profilereplyspeed'][i])
        else:
            profilereplyspeed.append(None)
    df_offer['profilereplyspeed'] = profilereplyspeed
    df_offer['profilereplyspeed'] = df_offer['profilereplyspeed'].astype(float)
    df_offer['profilereplyspeed'].fillna(df_offer['profilereplyspeed'].mean())


    # build profilefollowers binary
    profilefollowers = []
    for i in range(0, len(df_offer['profilefollowers'])):
        if len(df_offer['profilefollowers'][i]) > 0:
            profilefollowers.append(df_offer['profilefollowers'][i])
        else:
            profilefollowers.append(None)
    df_offer['profilefollowers'] = profilefollowers
    df_offer['profilefollowers'] = df_offer['profilefollowers'].astype(float)


    # build shipping binary
    shipping = []
    for i in df_offer['shippings']:
        if len(i) == 0:
            x = 1
            shipping.append(0)
        else:
            if len(i[0]) > 15:
                shipping.append(1)
            else:
                shipping.append(0)

    df_offer['shippings'] = shipping
    df_offer['shippings'] = df_offer['shippings'].astype(float)


    # location
    X = df_offer['locations'].astype(str)
    location = []
    for y in range(0, len(X)):
        if len(X[y]) > 4:
            location.append(X[y][0:5])
        else:
            location.append(None)
    df_offer['locations'] = location
    df_offer['locations'] = df_offer['locations'].astype(float)
    df_offer.rename(columns={'paymentdetails': 'Sicher_bezahlen', 'usertype': 'Gewerblicher_user', 'year': 'startyear', 'weekday': 'startweekday', 'month': 'startdayofmonth', 'dayofmonth': 'startmonth', 'scraptime': 'scrape_time', 'perception_descr': 'mistake_rate'}, inplace=True)

    return df_offer


def filledprofiledata(beautiful_df):

    beautiful_df['profilerating'] = beautiful_df['profilerating'].fillna(0)
    beautiful_df['profilefriendliness'] = beautiful_df['profilefriendliness'].fillna(0)
    beautiful_df['profilereliability'] = beautiful_df['profilereliability'].fillna(0)
    beautiful_df['profilereplyrate'] = beautiful_df['profilereplyrate'].fillna(0)
    beautiful_df['profilereplyspeed'] = beautiful_df['profilereplyspeed'].fillna(24)
    beautiful_df['profilefollowers'] = beautiful_df['profilefollowers'].fillna(0)

    return beautiful_df


def cleandata(more_beautiful_df, newest_offer):

    id = str(newest_offer).split('/')
    id = id[-1].split('-')
    new_list = []
    for i in range(0, len(more_beautiful_df['offer-ID'])):
        if len(more_beautiful_df['offer-ID'][i]) > 0:
            new = more_beautiful_df['offer-ID'][i]
            new_list.append(new[0])
        else:
            new_list.append(id[0])
    more_beautiful_df['offer-ID'] = new_list

    prices_list = []
    for sub in more_beautiful_df['prices']:
        sub = sub.replace(" VB", '')
        sub = sub.replace(" ", "")
        sub = sub.replace(".", "")
        sub = sub.replace("€", "")
        sub = sub.replace("VB", '')
        sub = sub.replace("Zuverschenken", '0')
        if sub == '':
            sub = '0'
        prices_list.append(sub)

    new_list = []
    for i in prices_list:
        i = i.split(' ')
        new_list.append(i[0])
    more_beautiful_df['prices'] = new_list

    new_list = []
    for i in more_beautiful_df['dates']:
        if len(i) > 0:
            new_list.append(i[0])
        else:
            new_list.append('18.12.2022')

    more_beautiful_df['dates'] = new_list

    del more_beautiful_df['views']

    more_beautiful_df['offer-ID'] = more_beautiful_df['offer-ID'].astype(int)
    more_beautiful_df['prices'] = more_beautiful_df['prices'].astype(int)
    more_beautiful_df['shippings'] = more_beautiful_df['shippings'].astype(int)
    more_beautiful_df['locations'] = more_beautiful_df['locations'].astype(int)
    more_beautiful_df['user-ID'] = more_beautiful_df['user-ID'].astype(int)
    more_beautiful_df['name'] = more_beautiful_df['name'].astype(int)
    more_beautiful_df['Sicher_bezahlen'] = more_beautiful_df['Sicher_bezahlen'].astype(int)
    more_beautiful_df['Gewerblicher_user'] = more_beautiful_df['Gewerblicher_user'].astype(int)
    more_beautiful_df['offeringsonline'] = more_beautiful_df['offeringsonline'].astype(int)
    more_beautiful_df['offeringssum'] = more_beautiful_df['offeringssum'].astype(int)
    more_beautiful_df['profilerating'] = more_beautiful_df['profilerating'].astype(int)
    more_beautiful_df['profilefriendliness'] = more_beautiful_df['profilefriendliness'].astype(int)
    more_beautiful_df['profilereliability'] = more_beautiful_df['profilereliability'].astype(int)
    more_beautiful_df['profilereplyrate'] = more_beautiful_df['profilereplyrate'].astype(int)
    more_beautiful_df['profilereplyspeed'] = more_beautiful_df['profilereplyspeed'].astype(int)
    more_beautiful_df['profilefollowers'] = more_beautiful_df['profilefollowers'].astype(int)
    with open('more_beautiful_df.textmate', 'w') as file:
        file.write(str(more_beautiful_df) + '\n')

    return more_beautiful_df


def filter_df_profiles_offers(df_profiles_offers):

    apple_offer_ids = []
    for i in df_profiles_offers['apple-offer-url']:
        i = i.split('/')
        i = i[-1]
        i = i.split('-')
        i = int(i[0])
        apple_offer_ids.append(i)
    new = df_profiles_offers.isin(apple_offer_ids)
    df_profiles_offers['apple_url'] = new['offer-ID']
    new_list = []
    for i in range(0, len(df_profiles_offers)):
        if df_profiles_offers['apple_url'][i] == True:
            new_list.append(1)
        else:
            new_list.append(0)
    df_profiles_offers['apple_url'] = new_list
    df_profiles_offers = df_profiles_offers.loc[df_profiles_offers['apple_url'] == 1]
    df_profiles_offers = df_profiles_offers.drop_duplicates(subset=['offer-ID'], keep='first')
    return df_profiles_offers


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
    df1 = df1

    df1 = df1.sort_values('PLZ', ascending=True)
    df1.fillna(method='ffill', inplace=True)
    df1.fillna(method='bfill', inplace=True)
    df1 = df1.loc[(df1['prices'] > 0) & (df1['prices'] < 2800) & (df1['PLZ'] < 99999)]
    df1 = df1.reset_index(drop=True)

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
            mainproductclass['iphone 14 pro max'] = 1
        elif 'phon' in i and '14 ' in i and 'pro' in i:
            mainproductclass['iphone 14 (no pro max)'] = 1
        elif 'phon' in i and '14 ' in i and 'plus' in i:
            mainproductclass['iphone 14 (no pro max)'] = 1
        elif 'phon' in i and '14 ' in i:
            mainproductclass['iphone 14 (no pro max)'] = 1

        elif 'phon' in i and '13 ' in i and 'max' in i and 'pro' in i:
            detailled_product_class.append('iphone 13 pro max')
            mainproductclass['iphone 13 pro max'] = 1
        elif 'phon' in i and '13 ' in i and 'pro' in i:
            mainproductclass['iphone 13 (no pro max)'] = 1
        elif 'phon' in i and '13 ' in i and 'mini' in i:
            mainproductclass['iphone 13 (no pro max)'] = 1
        elif 'phon' in i and '13 ' in i:
            mainproductclass['iphone 13 (no pro max)'] = 1


        elif 'phon' in i and 'xr ' in i:
            mainproductclass['old iphone'] = 1
        elif 'phon' in i and 'xs ' in i:
            mainproductclass['old iphone'] = 1
        elif 'phon' in i and ' x ' in i:
            mainproductclass['old iphone'] = 1

        elif 'phon' in i and '11 ' in i:
            mainproductclass['iphone 11'] = 1

        elif 'phon' in i and '7 ' in i and 'plus' in i:
            mainproductclass['old iphone'] = 1
        elif 'phon' in i and '7 ' in i:
            mainproductclass['old iphone'] = 1
        elif 'phon' in i and ' 8' in i and 'plus' in i:
            mainproductclass['old iphone'] = 1
        elif 'phon' in i and ' 8' in i:
            mainproductclass['old iphone'] = 1

        elif 'phon' in i and '12 ' in i:
            mainproductclass['iphone 12'] = 1

        elif 'phon' in i:
            mainproductclass['old iphone'] = 1

        # AIRPODS
        elif 'pod' in i and 'max' in i:
            mainproductclass['new airpods'] = 1
        elif 'pod' in i and 'pro' in i:
            mainproductclass['new airpods'] = 1
        elif 'pod' in i and '2' in i:
            mainproductclass['old airpods'] = 1
        elif 'pod' in i and '3' in i:
            mainproductclass['old airpods'] = 1

        # IPADS
        elif 'pad' in i and 'pro' in i:
            mainproductclass['ipad pro'] = 1
        elif 'pad' in i and 'mini' in i:
            mainproductclass['ipad mini'] = 1
        elif 'pad' in i and 'air' in i:
            mainproductclass['ipad air'] = 1

        # APPLE WATCHES
        elif 'watch' in i:
            mainproductclass['watch'] = 1

        # MACBOOKS
        elif 'book' in i and 'air' in i:
            mainproductclass['macbook air'] = 1
        elif 'book' in i and 'pro' in i:
            mainproductclass['macbook pro'] = 1

        # OTHER
        else:
            mainproductclass['others'] = 1
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

    return df1


def descriptions_analysis(apple_offers):
    #apple_offers = apple_offers.head(10)

    nlp = spacy.load("de_core_news_sm")
    nlp.add_pipe("emoji", first=True)
    df1 = pd.DataFrame()
    for m in range(0, len(apple_offers['descriptions'])):
        i = str(apple_offers['descriptions'][m])
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

    return df1


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
    east = []
    for x in df['BUNDESLAND']:
        x = x.replace('Baden-Württemberg', 'Baden_Wuerttemberg')
        new_bl.append(x)
        if 'Berlin' in x or 'Brandenburg' in x or 'Sachsen' in x or 'Sachsen-Anhalt' in x or 'Thüringen' in x or 'Mecklenburg-Vorpommern' in x :
            east.append(1)
        else:
            east.append(0)
    df['BUNDESLAND'] = new_bl
    df['east'] = east
    df['east'] = df['east'].astype(int)

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
    return df3


def profiles_analysis(apple_offers):

    df = pd.DataFrame()
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


main()