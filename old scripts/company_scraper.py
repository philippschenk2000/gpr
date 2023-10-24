# math & other stuff
import datetime as datetime
from wordcloud import WordCloud
import openai
from random import choice
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.firefox import GeckoDriverManager
from random import randint

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
    df_jobs = pd.read_csv('csv_data/linkedin_jobs 2.csv')
    df_descr = pd.read_csv('csv_data/job_descriptions_19_12_22.csv')
    df = pd.merge(df_jobs, df_descr, on='absolute_url', how='left')

    companies_list = ['FNZ Group', 'Carta', 'Moonfare', 'iCapital', 'fundsaccess', 'European Bank for Financial Services GmbH (ebase®)',
                      'portagon', '21finance', 'CAIS', 'Yieldstreet', 'Forge', 'EquityZen', 'TIFIN', 'TIFIN Wealth', 'Mercury', 'Walnut',
                      'Vestmark', 'Cadre', 'Crowdstreet', 'LEX Markets', 'Rally', 'Titan', 'PeerStreet', 'Sofi', 'InvestCloud, Inc.',
                      'Delio', 'Mipise', 'CrowdEngine', 'eueco GmbH', 'Betterfront', '+SUBSCRIBE®', 'LenderKit', 'DIAMOS AG', 'Fundment',
                      'BetaSmartz', 'Carbon Equity', 'Stableton Financial AG', 'Titanbay', 'eFront', 'Securitize', 'ScalingFunds','Addepar',
                      'Asset Class', 'Avaloq', 'Atominvest', 'Elinvar.de', 'Fincite', 'Ansarada', 'Global Shares', 'WEADVISE',
                      'SS&C Advent', 'White Label Crowdfunding Limited', 'fundsonchain', 'eFonds AG', 'CONDA Crowdinvesting',
                      'Magnifi by TIFIN', 'PM Alpha', 'AngelList', 'Wefunder', 'Allocations', 'Assure', 'Vestlane',
                      'Aduro Advisors', 'Backbase', 'Goji', 'Equi', 'Allfunds']


    firma = '21finance'
    #company_jobs2 = get_company_jobs2(agents, companies_list)
    df = renaming(df)
    results = analysis(df, firma)


    #descriptions_analysed = descriptions(df)
    # job_descriptions = get_job_descriptions(agents)



def get_company_jobs2(agents, companies_list):

    companies_linkedin = {'FNZ Group': '351913',
                          'Carta': '3163250',
                          'Moonfare': '11288180',
                          'iCapital': '3080201',
                          'fundsaccess': '6830490',
                          'European Bank for Financial Services GmbH (ebase®)': '516131',
                          'portagon': '10040565',
                          '21finance': '71227290',
                          'CAIS': '948215',
                          'Yieldstreet': '9383317',
                          'Forge': '3598625',
                          'EquityZen': '3188568',
                          'TIFIN': '51670856',
                          'TIFIN Wealth': '81675143',
                          'Mercury': '19107985',
                          'Walnut': '67760135',
                          'Vestmark': '22756',
                          'Cadre': '3338049',
                          'Crowdstreet': '2968859',
                          'LEX Markets': '12609089',
                          'Rally': '10946732',
                          'Titan': '11249478',
                          'PeerStreet': '3878267',
                          'Sofi': '2301992',
                          'InvestCloud, Inc.': '1275545',
                          'Delio': '10099447',
                          'Mipise': '2931413',
                          'CrowdEngine': '3595936',
                          'eueco GmbH': '6372597',
                          'Betterfront': '15835358',
                          '+SUBSCRIBE®': '10346870',
                          'LenderKit': '40791507',
                          'DIAMOS AG': '113026',
                          'Fundment': '10794930',
                          'BetaSmartz': '6378826',
                          'Carbon Equity': '65333096',
                          'Stableton Financial AG': '13004429',
                          'Titanbay': '64614833',
                          'eFront': '22306',
                          'Securitize': '18452792',
                          'ScalingFunds': '14837989',
                          'Addepar': '705598',
                          'Asset Class': '73958845',
                          'Avaloq': '22027',
                          'Atominvest': '17925906',
                          'Elinvar.de': '12905563',
                          'Fincite': '5375563',
                          'Ansarada': '486839',
                          'Global Shares': '34857',
                          'WEADVISE': '86853999',
                          'SS&C Advent': '4859',
                          'White Label Crowdfunding Limited': '5298314',
                          'fundsonchain': '76218975',
                          'eFonds AG': '10059934',
                          'CONDA Crowdinvesting': '2832940',
                          'Magnifi by TIFIN': '40708872',
                          'PM Alpha': '82742570',
                          'AngelList': '1806556',
                          'Wefunder': '3229238',
                          'Allocations': '67126109',
                          'Assure': '1119053',
                          'Vestlane': '87166974',
                          'Aduro Advisors': '3757480',
                          'Backbase': '20992',
                          'Goji': '10300138',
                          'Equi': '70533177',
                          'Allfunds': '27163023',
                          'Capterra': '99431'
                          }

    df = pd.DataFrame()
    df2 = pd.read_csv('csv_data/linkedin_jobs 2.csv')
    df3 = df2.loc[df2['scraped_week'] != datetime.datetime.now().strftime("%W")]
    df2 = df2.loc[df2['scraped_week'] == datetime.datetime.now().strftime("%W")]
    for y in range(0, len(companies_list)):
        if companies_list[y] not in df2['company']:
            c1 = companies_linkedin[str(companies_list[y])]
            user_agent = choice(agents)
            headers = {'User-Agent': user_agent}
            html = requests.get(url=i, headers=headers).text
            sel = Selector(text=html)

            absolute_urls1 = []
            html = html.split('linkedin.com/jobs/view')
            for i in html:
                i = i.split('?')
                if len(i) > 0:
                    absolute_urls1.append('https://www.linkedin.com/jobs/view' + i[0])
            absolute_urls1 = absolute_urls1[1:]
            print(absolute_urls1)

            titel = sel.css('h3.base-search-card__title::text').extract()
            name = []
            if len(titel) > 0:
                for i in titel:
                    i = i.replace('\n', '')
                    i = i.replace('  ', '')
                    name.append(i)
            print(name)

            titel = sel.css('a.hidden-nested-link::text').extract()
            company = []
            if len(titel) > 0:
                for i in titel:
                    i = i.replace('\n', '')
                    i = i.replace('  ', '')
                    company.append(i)
            print(company)

            titel = sel.css('span.job-search-card__location::text').extract()
            location = []
            if len(titel) > 0:
                for i in titel:
                    i = i.replace('\n', '')
                    i = i.replace('  ', '')
                    location.append(i)
            print(location)

            titel = sel.css('time.job-search-card__listdate::attr(datetime)').extract()
            titel2 = sel.css('time.job-search-card__listdate--new::attr(datetime)').extract()
            titel = titel + titel2
            time = []
            if len(titel) > 0:
                for i in titel:
                    i = i.replace('\n', '')
                    i = i.replace('  ', '')
                    time.append(i)
            print(time)

            df1 = pd.DataFrame()
            df1['absolute_url'] = absolute_urls1
            df1['name'] = name
            df1['company'] = company
            df1['location'] = location
            df1['time_published'] = time
            df1['scraped_week'] = datetime.datetime.now().strftime("%W")
            df1['scraped_date'] = datetime.datetime.now().strftime("%d-%m-%y")

            df = df3.append(df1, ignore_index=True)
            df = df.drop_duplicates(subset=['absolute_url'], keep='first')
            df.to_csv('csv_data/linkedin_jobs 2.csv', index=False)
            df3 = df
            sleeptime = float(randint(1, 1))
            sleep(sleeptime)

    return df


def renaming(df):

    location = []
    lat = []
    lng = []
    for i in df['location']:
        i = i.lower()
        if 'german' in i:
            i = 'GERMANY'
            lat.append(480)
            lng.append(1135)
        elif 'singap' in i:
            i = 'SINGAPORE'
            lat.append(860)
            lng.append(1775)
        elif 'austri' in i:
            i = 'AUSTRIA'
            lat.append(518)
            lng.append(1145)
        elif 'ireland' in i:
            i = 'IRELAND'
            lat.append(463)
            lng.append(1022)
        elif 'czechia' in i:
            i = 'CZECHIA'
            lat.append(500)
            lng.append(1175)
        elif 'switzer' in i:
            i = 'SWITZERLAND'
            lat.append(520)
            lng.append(1120)
        elif 'austral' in i:
            i = 'AUSTRALIA'
            lat.append(1060)
            lng.append(2000)
        elif 'netherlands' in i:
            i = 'NETHERLANDS'
            lat.append(475)
            lng.append(1100)
        elif 'portugal' in i:
            i = 'PORTUGAL'
            lat.append(585)
            lng.append(1020)
        elif 'india' in i:
            i = 'INDIA'
            lat.append(1580)
            lng.append(725)
        elif 'london' in i or 'kingdom' in i:
            i = 'UK'
            lat.append(475)
            lng.append(1065)
        elif 'canada' in i or 'toronto' in i:
            i = 'CANADA'
            lat.append(350)
            lng.append(450)
        elif 'new york' in i or 'santa clara' in i or 'francisco' in i or 'jersey' in i or 'states' in i or 'salt lake' in i or 'chicago' in i or 'wakefield' in i or 'jacksonville' in i or 'cottonwood' in i or 'boulder' in i or 'washington' in i or 'hamilton' in i or 'miami' in i or 'sacramento' in i or 'trenton' in i or 'atlanta' in i or 'hartford' in i or 'austin' in i or 'springfield' in i or 'birmingham' in i or 'greenwich' in i or 'portland' in i or 'claymont' in i or 'frisco' in i or 'hollywood' in i or 'tampa' in i or 'warren' in i or 'angeles' in i or 'san diego' in i or 'boston' in i or 'mountain view' in i or 'nashville' in i or 'denver' in i or 'boise' in i:
            i = 'USA'
            lat.append(530)
            lng.append(530)
        elif 'remote' in i:
            i = 'REMOTE'
            lat.append(-30)
            lng.append(-30)
        else:
            i = 'OTHER'
            lat.append(-30)
            lng.append(-30)
        location.append(i)
    df['lat'] = lat
    df['lng'] = lng


    industry = []
    for i in df['name']:
        i = i.lower()
        if 'product' in i or 'devop' in i or 'platform' in i or 'design' in i or 'project' in i or 'configuration' in i:
            i = 'PRODUCT'
        elif 'marketing' in i or 'brand' in i or 'demand gener' in i:
            i = 'MARKETING'
        elif 'data' in i or 'controller' in i:
            i = 'DATA'
        elif 'compliance' in i or 'legal' in i or 'jurist' in i or 'regulat' in i or 'investigation' in i or 'aml' in i:
            i = 'COMPLIANCE'
        elif 'financ' in i or 'business' in i or 'accounting' in i or 'accountant' in i or 'audit' in i or 'risk' in i or 'tax' in i or 'invest' in i or 'risiko' in i or 'bankkauf' in i or 'banking' in i or 'expansion' in i:
            i = 'FINANCE'
        elif 'sales' in i:
            i = 'SALES'
        elif 'client' in i or 'investor' in i or 'relationshi' in i or 'suppor' in i or 'contact' in i or 'partnership' in i or 'customer' in i:
            i = 'RELATIONS'
        elif 'people' in i or 'human' in i or 'hr' in i or 'talent' in i or 'office' in i or 'recruit' in i or 'network' in i or 'compensation' in i or 'staff' in i:
            i = 'PEOPLE'
        elif 'engineer' in i or 'information' in i or 'develop' in i or 'audit' in i or 'technolo' in i or 'implement' in i or 'it' in i:
            i = 'ENGINEERING'
        else:
            i = 'OTHER'
        industry.append(i)


    year = []
    month = []
    monthday = []
    for i in df['time_published']:
        i = i.split('-')
        year.append(i[0])
        month.append(i[1])
        monthday.append(i[2][:2])

    df['location2'] = df['location']
    df['year'] = year
    df['month'] = month
    df['monthday'] = monthday
    df['location'] = location
    df['industry'] = industry
    #del df['time_published']

    return df


def analysis(df, firma):

    df1 = df.loc[df['month'] == '12']
    df11 = df.loc[df['month'] == '01']

    df1['monthday'] = df1['monthday'].astype(int)
    df1 = df1.sort_values('monthday', ascending=True)
    df11['monthday'] = df11['monthday'].astype(int)
    df11 = df11.sort_values('monthday', ascending=True)
    df1 = df1.append(df11, ignore_index=True)
    df2 = df['company'].value_counts().to_frame()
    df2 = df2.reset_index()
    df2.columns = ['company', 'count']
    df4 = df['location'].value_counts().to_frame()
    df4 = df4.reset_index()
    df4.columns = ['location', 'loc_count']
    df5 = pd.merge(df, df4, on='location', how='left')

    df6 = df
    df6 = df6.loc[(df6['location'] == 'GERMANY') | (df6['location'] == 'AUSTRIA') | (df6['location'] == 'IRELAND') | (df6['location'] == 'CZECHIA') | (df6['location'] == 'SWITZERLAND') | (df6['location'] == 'NETHERLANDS') | (df6['location'] == 'PORTUGAL') | (df6['location'] == 'FRANCE') | (df6['location'] == 'UK')]
    location = []
    lat = []
    lng = []
    for i in df6['location2']:
        i = i.lower()
        if 'frankfu' in i or 'berlin' in i or 'munich' in i or 'münchen' in i:
            i = 'GERMANY'
            lat.append(550)
            lng.append(950)
        elif 'vienna' in i or 'wien' in i:
            i = 'AUSTRIA'
            lat.append(740)
            lng.append(1055)
        elif 'dublin' in i:
            i = 'IRELAND'
            lat.append(360)
            lng.append(470)
        elif 'london' in i:
            i = 'UK'
            lat.append(470)
            lng.append(640)
        elif 'czechia' in i:
            i = 'CZECHIA'
            lat.append(645)
            lng.append(1070)
        elif 'lisbon' in i or 'lisboa' in i or 'lissabon' in i:
            i = 'PORTUGAL'
            lat.append(975)
            lng.append(215)
        elif 'switzerland' in i:
            i = 'SWITZERLAND'
            lat.append(760)
            lng.append(850)
        elif 'amsterdam' in i:
            i = 'NETHERLENDS'
            lat.append(490)
            lng.append(815)
        else:
            i = 'OTHER'
            lat.append(-30)
            lng.append(-30)
        location.append(i)
    df6['lat'] = lat
    df6['lng'] = lng
    df6['location2'] = location

    df7 = df6['location2'].value_counts().to_frame()
    df7 = df7.reset_index()
    df7.columns = ['location2', 'loc_count']
    df8 = pd.merge(df6, df7, on='location2', how='left')

    firma = df1.loc[(df1['company'] == str(firma))]
    firma = firma.reset_index()
    if len(firma) == 0:
        print('Noch gar keine Jobs für dieses Unternehmen')
    del firma['index']
    idx = pd.date_range("2022-12-01", periods=200, freq="D")

    ts = pd.Series(range(len(idx)), index=idx)
    dfxy = pd.DataFrame()
    dfxy['time_published'] = ts
    dfxy = dfxy.reset_index()
    dfxy.columns = ['time_published', 'count']
    del dfxy['count']
    dfxy['time_published'] = dfxy['time_published'].astype(str)
    df22 = firma['time_published'].value_counts().to_frame()
    df22 = df22.reset_index()
    df22.columns = ['time_published', 'count']
    df222 = pd.merge(dfxy, df22, on='time_published', how='left')
    df222 = pd.merge(df222, firma, on='time_published', how='left')
    df2222 = pd.merge(dfxy, df1, on='time_published', how='left')


    '''keywords = ' '.join(df6['keywords'])
    keywords = keywords.replace('[', '')
    keywords = keywords.replace(']', '')
    keywords = keywords.replace('the', '')
    keywords = keywords.replace("'", '')
    print(keywords)

    fig, ax = plt.subplots(figsize=(6, 6))
    wc = WordCloud(max_words=50, background_color="white").generate(keywords)
    plt.imshow(wc)
    plt.axis('off')
    plt.title("Keywords from Job Offerings")'''

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(data=df222, x='time_published', palette='icefire', binwidth=1, hue='industry', multiple="stack")
    if len(firma['company']) > 0:
        plt.title("Job Postings of " + str(firma['company'][0]) + " per Industry")
    plt.xticks(rotation=45)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(data=df222, x='time_published', palette='icefire', binwidth=1, hue='location', multiple="stack")
    if len(firma['company']) > 0:
        plt.title("Job Postings of " + str(firma['company'][0]) + " per Country")
    plt.xticks(rotation=45)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(data=df, hue="location", y="company", multiple="stack", palette='icefire')
    plt.title("Job Postings per Country and per Company")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(data=df, hue="industry", y="company", multiple="stack")
    plt.title("Job Postings per Industry and per Company")

    size = df8['loc_count']
    img = plt.imread('Bildschirmfoto 2022-12-23 um 00.50.54.png')
    fig, ax = plt.subplots(figsize=(12, 7))
    pal = sns.color_palette('viridis', n_colors=5, as_cmap=True)
    sns.scatterplot(x='lng', y='lat', s=(size ** 1) * 30, data=df8, palette='dark:#5A9_r', cmap='Blues', hue=size)
    sns.scatterplot(ax=ax)
    ax.imshow(img, aspect='auto')
    plt.title("Job Postings per Country (europe)")

    size = df5['loc_count']
    img = plt.imread('Bildschirmfoto 2022-12-20 um 23.36.24.png')
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(x='lng', y='lat', s=(size ** 0.6) * 30, data=df, palette='dark:#5A9_r', cmap='Blues', hue=size)
    sns.scatterplot(ax=ax)
    ax.imshow(img, aspect='auto')
    plt.title("Job Postings per Country (worldwide)")

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df2, y='company', x='count', color='orange')
    plt.title("Job Postings by Company")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(data=df2222, x='time_published', binwidth=1, hue='month', multiple='stack')
    plt.title("Recent Job Postings")
    plt.xticks(rotation=45)
    plt.show()
    #'''




def get_job_descriptions(agents):

    df = pd.read_csv('csv_data/jobs_19_12_22.csv')
    df3 = pd.read_csv('csv_data/job_descriptions_19_12_22.csv')
    all_descriptions = []
    for j in range(0, len(df)):
        description = []
        if df['company'][j] != 'lalabumbum' and str(df['absolute_url'][j]) not in df3['absolute_url']:
            print(df['absolute_url'][j])
            user_agent = choice(agents)
            headers = {'User-Agent': user_agent}
            html = requests.get(url=i, headers=headers).text

            sel = Selector(text=html)
            #print(html)


            des4 = []
            for x in range(1, 15):
                stelle5 = str('#content > p:nth-child({})::text').format(x)
                description5 = sel.css(str(stelle5)).extract()
                stelle4 = str('#content > ul:nth-child({})::text').format(x)
                description4 = sel.css(str(stelle4)).extract()
                stelle8 = str('p.p2:nth-child({})::text').format(x)
                description8 = sel.css(str(stelle8)).extract()
                stelle14 = str('#content > div:nth-child({})::text').format(x)
                description14 = sel.css(str(stelle14)).extract()
                stelle17 = str('#content-intro > p:nth-child({})::text').format(x)
                description17 = sel.css(str(stelle17)).extract()
                stelle18 = str('span.NormalTextRun:nth-child({})::text').format(x)
                description18 = sel.css(str(stelle18)).extract()
                description5 = ' '.join(description5)
                description4 = ' '.join(description4)
                description8 = ' '.join(description8)
                description14 = ' '.join(description14)
                description17 = ' '.join(description17)
                description18 = ' '.join(description18)
                des4.append(description5)
                des4.append(description4)
                des4.append(description8)
                des4.append(description14)
                des4.append(description17)
                des4.append(description18)
                des3 = []
                for y in range(1, 12):
                    stelle3 = str('#content > ul:nth-child({}) > li:nth-child({})::text').format(x, y)
                    description3 = sel.css(str(stelle3)).extract()
                    stelle7 = str('#content > p:nth-child({}) > span:nth-child({})::text').format(x, y)
                    description7 = sel.css(str(stelle7)).extract()
                    stelle10 = str('.content-conclusion > ul:nth-child({}) > li:nth-child({})::text').format(x, y)
                    description10 = sel.css(str(stelle10)).extract()
                    stelle20 = str('#content-intro > p:nth-child({}) > span:nth-child({})::text').format(x, y)
                    description20 = sel.css(str(stelle20)).extract()
                    stelle21 = str('div.section:nth-child({}) > div:nth-child({})::text').format(x, y)
                    description21 = sel.css(str(stelle21)).extract()
                    description3 = ' '.join(description3)
                    description7 = ' '.join(description7)
                    description10 = ' '.join(description10)
                    description20 = ' '.join(description20)
                    description21 = ' '.join(description21)
                    des3.append(description3)
                    des3.append(description7)
                    des3.append(description10)
                    des3.append(description20)
                    des3.append(description21)
                    des2 = []
                    for z in range(1, 12):
                        stelle2 = str('#content > ul:nth-child({}) > li:nth-child({}) > ul:nth-child({})::text').format(x, y, z)
                        description2 = sel.css(str(stelle2)).extract()
                        stelle6 = str('#content > ul:nth-child({}) > li:nth-child({}) > span:nth-child({})::text').format(x, y, z)
                        description6 = sel.css(str(stelle6)).extract()
                        stelle9 = str('ul.ul1:nth-child({}) > li:nth-child({}) > span:nth-child({})::text').format(x, y, z)
                        description9 = sel.css(str(stelle9)).extract()
                        stelle11 = str('#content > div:nth-child({}) > div:nth-child({}) > div:nth-child({})::text').format(x, y, z)
                        description11 = sel.css(str(stelle11)).extract()
                        stelle12 = str('#content > div:nth-child({}) > p:nth-child({}) > span:nth-child({})::text').format(x, y, z)
                        description12 = sel.css(str(stelle12)).extract()
                        stelle15 = str('#content > div:nth-child({}) > ul:nth-child({}) > li:nth-child({})::text').format(x, y, z)
                        description15 = sel.css(str(stelle15)).extract()
                        stelle16 = str('#content > div:nth-child({}) > div:nth-child({}) > p:nth-child({})::text').format(x, y, z)
                        description16 = sel.css(str(stelle16)).extract()
                        stelle19 = str('#content-intro > ul:nth-child({}) > li:nth-child({}) > span:nth-child({})::text').format(x, y, z)
                        description19 = sel.css(str(stelle19)).extract()
                        description2 = ' '.join(description2)
                        description6 = ' '.join(description6)
                        description9 = ' '.join(description9)
                        description11 = ' '.join(description11)
                        description12 = ' '.join(description12)
                        description15 = ' '.join(description15)
                        description16 = ' '.join(description16)
                        description19 = ' '.join(description19)
                        des2.append(description2)
                        des2.append(description6)
                        des2.append(description9)
                        des2.append(description11)
                        des2.append(description12)
                        des2.append(description15)
                        des2.append(description16)
                        des2.append(description19)
                        des1 = []
                        for zz in range(1, 10):
                            stelle1 = str('#content > ul:nth-child({}) > li:nth-child({}) > ul:nth-child({}) > li:nth-child({})::text').format(x, y, z, zz)
                            description1 = sel.css(str(stelle1)).extract()
                            stelle13 = str('#content > div:nth-child({}) > ul:nth-child({}) > li:nth-child({}) > span:nth-child({})::text').format(x, y, z, zz)
                            description13 = sel.css(str(stelle13)).extract()
                            stelle22 = str('#content > div:nth-child({}) > div:nth-child({}) > p:nth-child({}) > span:nth-child({})::text').format(x, y, z, zz)
                            description22 = sel.css(str(stelle22)).extract()
                            description1 = ' '.join(description1)
                            description13 = ' '.join(description13)
                            description22 = ' '.join(description22)
                            des1.append(description1)
                            des1.append(description13)
                            des1.append(description22)
                        des1 = ''.join(des1)
                        if des1 != '' or des1 != '\n':
                            des2.append(des1)
                    des2 = ''.join(des2)
                    if des2 != '' or des2 != '\n':
                        des3.append(des2)
                des3 = ''.join(des3)
                if des3 != '' or des3 != '\n':
                    des4.append(des3)
            des4 = ' '.join(des4)
            des4 = des4.replace('\n', '')
            des4 = des4.replace('  ', ' ')
            if des4 != '' or des4 != '\n':
                description.append(des4)
            description = ' '.join(description)
            #print([description])
            all_descriptions.append(description)

            df1 = pd.DataFrame()
            df1['absolute_url'] = [str(df['absolute_url'][j])]
            df1['description'] = [description]
            print(df1)

            df2 = pd.read_csv('csv_data/job_descriptions_19_12_22.csv')
            df2 = df2.append(df1, ignore_index=True)
            df2.to_csv('csv_data/job_descriptions_19_12_22.csv', index=False)
            sleeptime = float(randint(0, 0))
            sleep(sleeptime)

    return df


def descriptions(df):

    openai.api_key = OPENAI_API_KEY

    keywords_for_frame = []
    for i in df['description']:
        #'''
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt="Extract keywords from this text: " + i,
            temperature=0.5,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.8,
            presence_penalty=0.0
        )
        print(response)
        '''

        response = {"choices": [
                        {
                          "finish_reason": "length",
                          "index": 0,
                          "logprobs": 0,
                          "text": "\n\n-Black-on-black ware \n-Puebloan Native American ceramic artists \n-Northern New Mexico \n-Reduction fired blackware \n-Selective burnishing \n-Refractory slip \n-Carving/incising designs \n-K"
                        }
                      ],
                      "created": 1671566073,
                      "id": "cmpl-6Pcz3HAM8s1LaDIwGkaI7B2u7TT4f",
                      "model": "text-davinci-003",
                      "object": "text_completion",
                      "usage": {
                        "completion_tokens": 5,
                        "prompt_tokens": 187,
                        "total_tokens": 192
                      }
                    }
        #'''
        print(response)
        keywords = response['choices'][0]['text']
        keywords = keywords.split('\n')
        keyword_list = []
        for x in keywords:
            if x != '':
                keyword_list.append(x[1:-1])
        keywords_for_frame.append(keyword_list)

        sleeptime = float(randint(4, 8))
        sleep(sleeptime)
    df['keywords'] = keywords_for_frame

    df2 = pd.read_csv('csv_data/jobs_all.csv')
    df2 = df2.append(df, ignore_index=True)
    df2.to_csv('csv_data/jobs_all.csv', index=False)

    return df2



main()