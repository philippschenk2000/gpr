#from selenium import webdriver
import re
import pandas as pd
import csv
from random import randint
from time import sleep
import warnings
from webdriver_manager.firefox import GeckoDriverManager

warnings.filterwarnings("ignore")
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
#from webdriver_manager.chrome import ChromeDriverManager
#driver = webdriver.Chrome(ChromeDriverManager().install())
driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())


def main():

    ''' INPUT DATA '''
    data = pd.read_csv('/Users/philippschenk/PycharmProjects/facebook-comment-scraper-master/csv_data/gilette2.csv')
    folder = 'gilette2'
    ''' INPUT DATA '''


    # COOKIES
    driver.get(data['comment_url'][len(pd.read_csv('/Users/philippschenk/PycharmProjects/facebook-comment-scraper-master/checked_results_{}.csv'.format(folder)))])
    driver.implicitly_wait(4)
    print('Length goal: ' + str(len(data.drop_duplicates(subset=['comment_id'], keep='first'))))


    if 'posts' in driver.current_url:
        # COOKIES
        driver.find_elements_by_xpath("//button[contains(string(), 'Erforderliche und optionale Cookies erlauben')]")[0].click()
        # LOGIN
        email_element = driver.find_element_by_id('email')
        email_element.send_keys('hoyibit484@invodua.com')
        pw_element = driver.find_element_by_id('pass')
        pw_element.send_keys('sf55FFas')
        loginbutton_element = driver.find_element_by_id('loginbutton')
        loginbutton_element.click()

        # CONTENT
        print(driver.current_url)
        for j in range (0, len(data['comment_url']), 1):
            old_data = pd.read_csv('/Users/philippschenk/PycharmProjects/facebook-comment-scraper-master/checked_results_{}.csv'.format(folder))
            old_data['comment_id'] = old_data['comment_id'].astype(int)
            comment_url = data['comment_url'][j]
            comment_id = data['comment_id'][j].astype(int)
            new_df = old_data.loc[old_data['comment_id'] == comment_id]
            print(comment_url)

            if len(new_df) == 0:
                driver.get(comment_url)
                driver.implicitly_wait(4)
                source_code = str(driver.page_source)
                ids_little = []
                times_little = []
                #ids_little.append(comment_id)
                #times_little.append(0)
                for l in data['comment_id']:
                    search = '","legacy_fbid":"' + str(l)
                    if source_code.find(search) > 0:
                        ids_little.append(l)
                        new_source_code = source_code[source_code.find(search):source_code.find(search)+2000]
                        search2 = ',"created_time":'
                        new_source_code.find(search2)
                        time = new_source_code[new_source_code.find(search2)+16:new_source_code.find(search2)+26]
                        times_little.append(time)
                df = pd.DataFrame()
                df['comment_id'] = ids_little
                df['created_time'] = times_little
                df['created_time'] = df['created_time'].astype(int)
                print(len(df))
                df2 = pd.read_csv('/Users/philippschenk/PycharmProjects/facebook-comment-scraper-master/checked_results_{}.csv'.format(folder))
                df2 = df2.append(df, ignore_index=True)
                df2 = df2.drop_duplicates(subset=['comment_id'], keep='first')
                df2.to_csv('/Users/philippschenk/PycharmProjects/facebook-comment-scraper-master/checked_results_{}.csv'.format(folder), index=False)
                sleeptime = float(randint(17, 20))
                sleep(sleeptime)


    elif 'photos' in driver.current_url:
        driver.find_element_by_css_selector("div[aria-label='Nur erforderliche Cookies erlauben']").click()
        # LOGIN
        email_element = driver.find_element_by_name('email')
        email_element.send_keys('hoyibit484@invodua.com')
        pw_element = driver.find_element_by_name('pass')
        pw_element.send_keys('sf55FFas')
        loginbutton_element = driver.find_element_by_css_selector("div[aria-label='Accessible login button']")
        loginbutton_element.click()

        # CONTENT
        print(driver.current_url)
        for j in range(0, len(data['comment_url']), 1):
            old_data = pd.read_csv('/Users/philippschenk/PycharmProjects/facebook-comment-scraper-master/checked_results_{}.csv'.format(folder))
            old_data['comment_id'] = old_data['comment_id'].astype(int)
            comment_url = data['comment_url'][j]
            comment_id = data['comment_id'][j].astype(int)
            new_df = old_data.loc[old_data['comment_id'] == comment_id]
            print(comment_url)

            if len(new_df) == 0:
                driver.get(comment_url)
                driver.implicitly_wait(4)
                source_code = str(driver.page_source)
                #print(source_code)
                ids_little = []
                times_little = []
                #ids_little.append(comment_id)
                #times_little.append(0)
                #print(source_code[source_code.find('is_author_banned_by_content_owner:":false,"is_author_original_poster":false,"created_time":'):source_code.find('is_author_banned_by_content_owner:":false,"is_author_original_poster":false,"created_time":')+200])
                for l in data['comment_id']:
                    search = '","legacy_fbid":"' + str(l)
                    if source_code.find(search) > 0:
                        ids_little.append(l)
                        new_source_code = source_code[source_code.find(search):source_code.find(search) + 2000]
                        search2 = ',"created_time":'
                        new_source_code.find(search2)
                        time = new_source_code[new_source_code.find(search2) + 16:new_source_code.find(search2) + 26]
                        times_little.append(time)
                df = pd.DataFrame()
                df['comment_id'] = ids_little
                df['created_time'] = times_little
                df['created_time'] = df['created_time'].astype(int)
                print(len(df))
                df2 = pd.read_csv( '/Users/philippschenk/PycharmProjects/facebook-comment-scraper-master/checked_results_{}.csv'.format(folder))
                df2 = df2.append(df, ignore_index=True)
                df2 = df2.drop_duplicates(subset=['comment_id'], keep='first')
                df2.to_csv('/Users/philippschenk/PycharmProjects/facebook-comment-scraper-master/checked_results_{}.csv'.format(folder), index=False)
                sleeptime = float(randint(17, 20))
                sleep(sleeptime)


    elif 'videos' in driver.current_url:
        driver.find_element_by_css_selector("div[aria-label='Nur erforderliche Cookies erlauben']").click()
        # LOGIN
        email_element = driver.find_element_by_name('email')
        email_element.send_keys('hoyibit484@invodua.com')
        pw_element = driver.find_element_by_name('pass')
        pw_element.send_keys('sf55FFas')
        loginbutton_element = driver.find_element_by_css_selector("div[aria-label='Accessible login button']")
        loginbutton_element.click()

        # CONTENT
        print(driver.current_url)
        for j in range(0, len(data['comment_url']), 1):
            old_data = pd.read_csv('/Users/philippschenk/PycharmProjects/facebook-comment-scraper-master/checked_results_{}.csv'.format(folder))
            old_data['comment_id'] = old_data['comment_id'].astype(int)
            comment_url = data['comment_url'][j]
            comment_id = data['comment_id'][j].astype(int)
            new_df = old_data.loc[old_data['comment_id'] == comment_id]
            print(comment_url)

            if len(new_df) == 0:
                driver.get(comment_url)
                driver.implicitly_wait(4)
                source_code = str(driver.page_source)
                ids_little = []
                times_little = []
                #ids_little.append(comment_id)
                #times_little.append(0)
                for l in data['comment_id']:
                    search = '","legacy_fbid":"' + str(l)
                    if source_code.find(search) > 0:
                        ids_little.append(l)
                        new_source_code = source_code[source_code.find(search):source_code.find(search) + 2000]
                        search2 = ',"created_time":'
                        new_source_code.find(search2)
                        time = new_source_code[new_source_code.find(search2) + 16:new_source_code.find(search2) + 26]
                        times_little.append(time)
                df = pd.DataFrame()
                df['comment_id'] = ids_little
                df['created_time'] = times_little
                df['created_time'] = df['created_time'].astype(int)
                print(len(df))
                df2 = pd.read_csv( '/Users/philippschenk/PycharmProjects/facebook-comment-scraper-master/checked_results_{}.csv'.format(folder))
                df2 = df2.append(df, ignore_index=True)
                df2 = df2.drop_duplicates(subset=['comment_id'], keep='first')
                df2.to_csv('/Users/philippschenk/PycharmProjects/facebook-comment-scraper-master/checked_results_{}.csv'.format(folder), index=False)
                sleeptime = float(randint(17, 20))
                sleep(sleeptime)




main()
