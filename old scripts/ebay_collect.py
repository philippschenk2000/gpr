import datetime as datetime
import re
import warnings
from random import choice
from random import randint
from time import sleep
import pandas as pd
import requests
from requests.exceptions import ChunkedEncodingError
from scrapy import Selector


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

    start_page = 1
    end_page = 12

    urls = []
    for i in range(0, 1):
        for i in range(start_page, end_page+1):

        profileurls = []
        for j in urls:
            user_agent = choice(agents)
            headers = {'User-Agent': user_agent}
            trycnt = 5  # max try cnt
            while trycnt > 0:
                try:
                    html = requests.get(url=j, headers=headers).content
                    sel = Selector(text=html)
                    getofferurls = getofferurl(sel)
                    print('Die Anzeigen und deren dazugehörigen User werden sich von folgender Seite angeschaut:')
                    print(j)

                    for i in getofferurls:
                        newest_offer = i
                        html = requests.get(url=i, headers=headers).content
                        sel = Selector(text=html)
                        profiles = sel.css('html body#vap div.site-base div.site-base--content div#site-content.l-page-wrapper.l-container-row section#viewad-main.l-container-row section#viewad-cntnt.l-row.a-double-margin.l-container-row aside#viewad-sidebar.a-span-8.l-col div#viewad-profile-box.l-container-row.contentbox--vip.no-shadow.j-sidebar-content div#viewad-contact div.l-container-row ul.iconlist li span.iconlist-text span.text-body-regular-strong.text-force-linebreak a::attr(href)').extract()
                        if len(profiles) > 0:
                            profileurls.append(y)

                            for f in range(0, 1):
                                i = y
                                user_agent = choice(agents)
                                headers = {'User-Agent': user_agent}
                                html = requests.get(url=i, headers=headers).content
                                sel = Selector(text=html)
                                df = getprofiles(sel, i)
                                df_offer = profileswithoffers(df, headers, newest_offer)
                                beautiful_df = editprofiledata(df_offer)
                                more_beautiful_df = filledprofiledata(beautiful_df)
                                most_beautiful_df = cleandata(more_beautiful_df, newest_offer)

                                df_just_profiles = pd.read_csv('csv_data/profiles.csv')
                                df_just_profiles = df_just_profiles.append(df, ignore_index=True)
                                df_just_profiles.to_csv('csv_data/profiles.csv', index=False)

                                df_profiles_offers = pd.read_csv('csv_data/profiles_offers.csv')
                                df_profiles_offers = df_profiles_offers.append(more_beautiful_df, ignore_index=True)
                                df_profiles_offers.to_csv('csv_data/profiles_offers.csv', index=False)

                                df_profiles_offers = pd.read_csv('csv_data/profiles_offers_clean.csv')
                                df_profiles_offers = df_profiles_offers.append(most_beautiful_df, ignore_index=True)
                                df_profiles_offers.to_csv('csv_data/profiles_offers_clean.csv', index=False)

                                all_offers_df = filter_df_profiles_offers(df_profiles_offers)
                                df_profiles_offers = pd.read_csv('csv_data/profiles_offers_apple_clean.csv')
                                df_profiles_offers = df_profiles_offers.iloc[0:0]
                                all_offers_df.to_csv('csv_data/profiles_offers_apple_clean.csv', index=False)

                                '''del all_offers_df['views']; del all_offers_df['name']; del all_offers_df['apple_url']
                                corr = all_offers_df.corr(method='pearson')
                                plt.figure(figsize=(20, 6))
                                sns.heatmap(corr, annot=True, cmap='Blues')
                                plt.title('Correlation matrix')
                                plt.show()'''

                except ChunkedEncodingError as ex:
                    if trycnt <= 0:
                        print("Failed to retrieve: " + j + "\n" + str(ex))  # done retrying
                    else:
                        trycnt -= 1  # retry
                    sleeptime = float(randint(1, 2))
                    sleep(sleeptime)
    # '''


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


main()