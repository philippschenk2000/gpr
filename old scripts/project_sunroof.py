import seaborn as sns
import warnings
import re
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.messagebox import showinfo
import matplotlib
from ttkwidgets.autocomplete import AutocompleteCombobox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import ImageTk,Image
matplotlib.use('TkAgg')


def getdata():
    laender = ['Austria', 'Belgium', 'Bulgaria', 'Switzerland', 'Cyprus', 'Czechia', 'Germany', 'Denmark', 'Estonia', 'Greece', 'Spain', 'Finland', 'France', 'Hungary', 'Croatia', 'Ireland', 'Italy', 'Luxembourg', 'Liechtenstein', 'Latvia', 'Lithuania', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania', 'Sweden', 'Slovenia', 'Slovakia', 'Turkey', 'United Kingdom']
    ausrichtung = ['flat', 'north', 'east', 'south', 'west']
    warnings.filterwarnings("ignore")

    def login_clicked():
        global plz, plzz, land, land1, neigung, neigung1, ausricht, ausricht1, surface, surface1, surface2, surface3, usage, usage1
        plzz = plz.get()
        land = land1.get()
        neigung = neigung1.get()
        ausricht = ausricht1.get()
        surface = surface1.get()
        surface2 = surface3.get()
        usage = usage1.get()
        msg = f' Das Solarpotential in: {land1.get()} an der PLZ: {plz.get()} wird berechnet'
        showinfo(title='Information', message=msg)
        land = land_change(land)
        neigungs_faktor = neigungs_rechner(neigung, ausricht, land)
        LAND = str(land)
        PLZ = str(plzz)
        LAND_PLZ = str(LAND) + str(PLZ)
        main1 = main(LAND_PLZ, PLZ, LAND, laender)
        x = main1['kwh/kwp_real'].astype(float)

        for i in range(0, 1):
            lat = main1['LAT'].astype(float)
            #main1['PRICE'] = [0.44]
            price = main1['PRICE'].astype(float)
            stadt = main1['STADT'].astype(str)
            bundesland = main1['BUNDESLAND'].astype(str)
            lng = main1['LNG'].astype(str)


            x = round((x + lat * neigungs_faktor * 30), 1)
            x = x.reset_index()
            x = x[0]
            global yyyy
            r1 = x.astype(str)
            r1 = r1[0]
            yyyy = r1
            result1.delete(0, 200)
            result1.insert(0, str(r1) + ' kWh/kWp')

            r3 = round(x * float(surface) * float(surface2) * 0.164, 1)
            r3 = r3.astype(str)
            r3 = r3[0]
            result3.delete(0, 200)
            result3.insert(0, str(r3) + ' kWh/roof (' + str(int(surface) * int(surface2)) + ' m2)')

            r5 = round(price * float(usage) * 20, 1)
            r5 = str(r5)
            r5 = re.split('[ ]', r5)
            r55 = []
            for x in r5:
                if x != '':
                    r55.append(x.replace("\nName:", ""))
            r5 = r55[1]
            result5.delete(0, 200)
            result5.insert(0,  '-' + str(r5) + ' EUR')

            kWp = float(surface) * float(surface2) * 0.164
            if kWp < 10:
                ertrag = 0.082 * kWp
            elif kWp < 40:
                ertrag = 0.082 * 10 + 0.075 * (kWp-10)
            else:
                ertrag = 0.082 * 10 + 0.075 * 30 + (kWp - 40) * 0.062
            r7 = round(ertrag/kWp * float(r3) * 20 * (1-float(usage)/float(r3)), 1)
            r7 = str(r7)
            result7.delete(0, 200)
            result7.insert(0, str(r7) + ' EUR (' + str(100-round((float(usage) / float(r3)) * 100, 2)) + '% Sale)')

            eigenverbrauch = float(price) * (float(usage))
            summe_eigenverbrauch = 0
            for i in range(0, 20):
                eigenverbrauch = eigenverbrauch * 1.02
                summe_eigenverbrauch = summe_eigenverbrauch + eigenverbrauch
            r77 = round(summe_eigenverbrauch, 1)
            r77 = str(r77)
            result77.delete(0, 200)
            result77.insert(0, str(r77) + ' EUR (' + str(round((float(usage) / float(r3)) * 100, 2)) + '% Use)')

            r9 = round(float(r77) + float(r7) - float(r5), 1)
            result9.delete(0, 200)
            result9.insert(0,str(r9) + ' EUR')

        fig3 = payout(LAND, x, main1)


    def payout(LAND, x, main1):
        figure = plt.Figure(figsize=(6, 3.5), facecolor='#FFFFCC', dpi=100)
        figure.clf(figure)
        figure.clear(figure)
        plt.close(figure)
        scatter = FigureCanvasTkAgg(figure, root)
        scatter.get_tk_widget().pack(side=tk.BOTTOM, pady=(20, 290))
        df = pd.read_csv('csv_data/EMHIRESPV_TSh_CF_Country_19862015.csv')
        df['Date'] = pd.DatetimeIndex(
            pd.date_range('1-1-1986T00:00:00', end='31-12-15T23:00:00', freq='H'))  # 403738318U
        df['month'] = df['Date'].apply(lambda x: x.month)
        df['day'] = df['Date'].apply(lambda x: x.day)
        df['hour'] = df['Date'].apply(lambda x: x.hour)
        df2 = pd.DataFrame()
        df2['payout'] = df[str(LAND)]

        df2['month'] = df['month']
        df2['day'] = df['day']
        df2['hour'] = df['hour']
        df2 = df2[-8760 * 1:]
        df2['payout'] = df2['payout'] * 0.81

        ax1 = figure.add_subplot(211)
        df3 = df2[["month", "payout"]].groupby(['month'], as_index=False).sum().sort_values(by='month',ascending=True)
        factor = float(yyyy) / main1['kwh/kwp_real'].astype(float)
        df3['payout'] = df3['payout'] * float(factor)
        ax1.plot(df3['month'], df3['payout'], color='orange')
        ax1.set_title('Production (per month & hour)')
        ax1.set_ylabel('kwh/kwp: '+ str(LAND))
        #ax1.cla()

        ax3 = figure.add_subplot(212)
        df4 = df2[["hour", "payout"]].groupby(['hour'], as_index=False).sum().sort_values(by='hour', ascending=True)
        factor = float(yyyy) / main1['kwh/kwp_real'].astype(float)
        df4['payout'] = df4['payout'] * float(factor)
        ax3.plot(df4['hour'], df4['payout'], color='orange')
        ax3.set_xlabel('hour of day')
        ax3.set_ylabel('kwh/kwp: '+ str(LAND))


        return figure


    for i in range(0, 1):
        root = tk.Tk()
        background = '#FFFFCC'
        foreground = '#000000'
        cellcolour = '#FFDF00'
        background2 = '#FFDF00'
        foreground2 = '#000000'
        fontsize = 12
        root.geometry('1920x1080')
        root.resizable(True, True)
        root.title('Your Solar Savings Calculator')
        root['bg'] = background

        global land, land1, plz, plzz, neigung, neigung1, ausricht, ausricht1, surface, surface1, surface2, surface3, usage, usage1
        x1 = 50
        x2 = 290
        y1 = 70
        y2 = 45
        y3 = 70

        schrift = tk.Label(text="Here you can calculate the profitability of a PV on your roof!", font=("Helvetica", 13, 'bold'), bg=background, fg=foreground)
        schrift.place(x=x1, y=24)

        land = tk.Label(text="Enter Country", font=("Helvetica", fontsize), bg=background, fg=foreground)
        land.place(x=x2, y=y3 + y2*0)
        land1 = AutocompleteCombobox(root, completevalues=laender, foreground='white', width=22, state='readonly')
        land1.place(x=x1, y=y1 + y2*0)

        '''im1 = Image.open('pics/png-transparent-european-union-flag-of-europe-general-data-protection-regulation-flag-miscellaneous-flag-logo.png')
        ph1 = im1.resize((40, 20), Image.ANTIALIAS)
        ph1 = ImageTk.PhotoImage(ph1)
        label1 = tk.Label(root, image=ph1, width=20, height=20)
        label1.place(x=x2-20, y=y3 + y2*0)'''


        plzs = tk.Label(text="Enter PLZ (60528, BT4, 0793)", font=("Helvetica", fontsize), fg=foreground, bg=background)
        plzs.place(x=x2, y=y3 + y2*1)
        plz = tk.StringVar()
        plz = tk.Entry(root, textvariable=plz, fg=foreground, width=24, bg=cellcolour)
        plz.place(x=x1, y=y1 + y2*1)
        plzz = plz.get()
        plz.insert(0, str(60528))

        ausricht = tk.Label(root, text="Enter roof position", font=("Helvetica", fontsize), bg=background, fg=foreground)
        ausricht.place(x=x2, y=y3 + y2*2)
        ausricht1 = AutocompleteCombobox(root, completevalues=ausrichtung, background='white', foreground='white', width=22, state='readonly')
        ausricht1.place(x=x1, y=y1 + y2*2)

        neigung = tk.Label(text="Enter roof tilt, in degrees", font=("Helvetica", fontsize), bg=background, fg=foreground)
        neigung.place(x=x2, y=y3 + y2*3)
        neigung1 = tk.StringVar()
        neigung1 = tk.Entry(root, textvariable=neigung1, fg=foreground, width=24, bg=cellcolour)
        neigung1.place(x=x1, y=y1 + y2*3)
        neigung1.insert(0, str(0))

        surface = tk.Label(text="Enter roof width, in m", font=("Helvetica", fontsize), bg=background, fg=foreground)
        surface.place(x=x2, y=y3 + y2*4)
        surface1 = tk.StringVar()
        surface1 = tk.Entry(root, textvariable=surface1, fg=foreground, width=24, bg=cellcolour)
        surface1.place(x=x1, y=y1 + y2*4)
        surface1.insert(0, str(10))

        surface2 = tk.Label(text="Enter roof length, in m", font=("Helvetica", fontsize), bg=background, fg=foreground)
        surface2.place(x=x2, y=y3 + y2*5)
        surface3 = tk.StringVar()
        surface3 = tk.Entry(root, textvariable=surface3, fg=foreground, width=24, bg=cellcolour)
        surface3.place(x=x1, y=y1 + y2*5)
        surface3.insert(0, str(10))

        usage = tk.Label(text="Enter yearly energy use, in kWh", font=("Helvetica", fontsize), bg=background, fg=foreground)
        usage.place(x=x2, y=y3 + y2*6)
        usage1 = tk.StringVar()
        usage1 = tk.Entry(root, textvariable=usage1, fg=foreground, width=24, bg=cellcolour)
        usage1.place(x=x1, y=y1 + y2*6)
        usage1.insert(0, str(4000))

        im = Image.open('pics/Bildschirmfoto 2022-11-24 um 18.51.28.png')
        ph = im.resize((500, 330), Image.ANTIALIAS)
        ph = ImageTk.PhotoImage(ph)
        label = tk.Label(root, image=ph, width=500, height=330)
        label.pack(side=tk.TOP, pady=(55, 0))



        result = tk.Label(text="Energy generation, per Year", font=("Helvetica", fontsize), bg=background, fg=foreground)
        result.place(x=x2, y=y3 + y2*9)
        result1 = tk.StringVar()  # plz_choice1
        result1 = tk.Entry(textvariable=result1, background='white', foreground='yellow', width=24, bg=background2, fg=foreground2)
        result1.place(x=x1, y=y1 + y2*9)

        result2 = tk.Label(text="Energy generation, per Year", font=("Helvetica", fontsize), bg=background, fg=foreground)
        result2.place(x=x2, y=y3 + y2*10)
        result3 = tk.StringVar()  # plz_choice1
        result3 = tk.Entry(textvariable=result3, background='white', foreground='yellow', width=24, bg=background2, fg=foreground2)
        result3.place(x=x1, y=y1 + y2*10)

        result4 = tk.Label(text="Total 20-year cost without solar (-)", font=("Helvetica", fontsize), bg=background, fg=foreground)
        result4.place(x=x2, y=y3 + y2*13.5)
        result5 = tk.StringVar()  # plz_choice1
        result5 = tk.Entry(textvariable=result5, background='white', foreground='orange', width=24, bg=background2, fg=foreground2)
        result5.place(x=x1, y=y1 + y2*13.5)

        result6 = tk.Label(text="Total 20-year return for prod. energy (+)", font=("Helvetica", fontsize), bg=background, fg=foreground)
        result6.place(x=x2, y=y3 + y2*11.5)
        result7 = tk.StringVar()  # plz_choice1
        result7 = tk.Entry(textvariable=result7, background='white', foreground='green', width=24, bg=background2, fg=foreground2)
        result7.place(x=x1, y=y1 + y2*11.5)

        result66 = tk.Label(text="Total 20-year savings for own use (+)", font=("Helvetica", fontsize), bg=background, fg=foreground)
        result66.place(x=x2, y=y3 + y2*12.5)
        result77 = tk.StringVar()  # plz_choice1
        result77 = tk.Entry(textvariable=result77, background='white', foreground='green', width=24, bg=background2, fg=foreground2)
        result77.place(x=x1, y=y1 + y2*12.5)

        result8 = tk.Label(text="Total return, excl. installation costs (=)", font=("Helvetica", fontsize, 'bold'), bg=background, fg=foreground)
        result8.place(x=x2, y=y3 + y2*14.7)
        result9 = tk.StringVar()  # plz_choice1
        result9 = tk.Entry(textvariable=result9, background='white', foreground='yellow', width=24, bg=background2, fg=foreground2)
        result9.place(x=x1, y=y1 + y2*14.7)

        canvas = tk.Canvas(width=470, height=0.00001)
        canvas.create_line(0, 0, 470, 0, width=0.00001)
        canvas.place(x=x1, y=y1 + y2*14.35)

        login_button = tk.Button(root, text="Calculate Solar Potential", font=("Helvetica", 16), command=login_clicked,
                                 fg='#000000', bg='#FF8C00')
        login_button.place(x=x1, y=y1 + y2 * 8 - 15)

        root.mainloop()


def land_change(land):

    #laender = ['AT', 'BE', 'BG', 'CH', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 'FR', 'HU', 'IE', 'IT', 'LU', 'LV', 'LT', 'NL', 'NO', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK', 'UK']
    #laender = ['Austria', 'Slovenia', 'Belgium', 'Bulgaria', 'Switzerland', 'Cyprus', 'Czechia', 'Germany', 'Denmark', 'Estonia', 'Greece', 'Spain', 'Finland', 'France', 'Hungary', 'Ireland', 'Italy', 'Luxembourg', 'Latvia', 'Lithuania', 'Netherlands', 'Norway', 'Poland','Portugal', 'Romania', 'Sweden', 'Slovenia', 'Slovakia', 'United Kingdom']
    land = land.replace('Austria', 'AT')
    land = land.replace('Belgium', 'BE')
    land = land.replace('Bulgaria', 'BG')
    land = land.replace('Switzerland', 'CH')
    land = land.replace('Cyprus', 'CY')
    land = land.replace('Czechia', 'CZ')
    land = land.replace('Germany', 'DE')
    land = land.replace('Denmark', 'DK')
    land = land.replace('Estonia', 'EE')
    land = land.replace('Greece', 'EL')
    land = land.replace('Spain', 'ES')
    land = land.replace('Finland', 'FI')
    land = land.replace('France', 'FR')
    land = land.replace('Hungary', 'HU')
    land = land.replace('Croatia', 'HR')
    land = land.replace('Ireland', 'IE')
    land = land.replace('Italy', 'IT')
    land = land.replace('Luxembourg', 'LU')
    land = land.replace('Latvia', 'LV')
    land = land.replace('Liechtenstein', 'LI')
    land = land.replace('Lithuania', 'LT')
    land = land.replace('Netherlands', 'NL')
    land = land.replace('Norway', 'NO')
    land = land.replace('Poland', 'PL')
    land = land.replace('Portugal', 'PT')
    land = land.replace('Romania', 'RO')
    land = land.replace('Sweden', 'SE')
    land = land.replace('Slovenia', 'SI')
    land = land.replace('Slovakia', 'SK')
    land = land.replace('Turkey', 'TR')
    land = land.replace('United Kingdom', 'UK')

    return land


def neigungs_rechner(neigung, ausricht, land):

    df = pd.read_csv('csv_data/europe.csv')
    df = df.loc[df['LAND'] == land]
    lat = df['LAT'].astype(float)

    a = (sum(lat) / len(lat))
    x = 0
    if neigung != '':
        x = int(neigung)
    y = 0
    if ausricht == 'flat':
        y = 0
    if x > 0:
        if ausricht == 'south':
            if x < a/2:
                y = 0
            else:
                y = -0.0000095*(x-a/2)**2.5
        if ausricht == 'west':
            y = -0.00001*x**2.5
        if ausricht == 'east':
            y = -0.00001*x**2.5
        if ausricht == 'north':
            y = -0.00002*x**2.5

    return y


def main(LAND_PLZ, PLZ, LAND, laender):

    warnings.filterwarnings("ignore")
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    #'''
    europe = knearest(LAND_PLZ, PLZ, LAND)
    europe['kwh/kwp_real'] = (europe['SUN1'] /3 ) *0.81 # Werte sind schließlich von < 2016
    europe['kwh/kwp'] = europe['SUN1']/(3) # ich habe ja 3 Jahre aufsummiert in sun1
    europe['google'] = europe['kwh/kwp'] * 0.75 # google geht von 75% sonnenstrahlung mindestens aus
    german_data3 = german_dataset3(europe, LAND_PLZ)
    #correl = correl1(europe, LAND_PLZ, LAND)
    #'''

    '''
    german_data = german_dataset(laender, LAND)
    german_data2 = german_dataset2(german_data, PLZ, LAND_PLZ)
    correl = correl2(german_data2)
    #'''

    return german_data3


def knearest(LAND_PLZ, PLZ, LAND):

    data = pd.read_csv('csv_data/ALL_EU_states.csv')
    data1 = data.loc[data['LAND_PLZ'] == LAND_PLZ]
    data2 = data.loc[data['LAND'] == LAND]
    data2['PLZ'] = data2['PLZ'].astype(str)

    if len(data1) < 1:
        data2 = data2.append({
            'LAND': LAND, 'PLZ': str(PLZ), 'STADT': None, 'BUNDESLAND': None, 'LAT': None, 'LNG': None, 'LAND_PLZ': LAND_PLZ, 'NUTS3': None, 'NUTS2': None,
            'NUTS1': None, 'NAME': None, 'CCODE': None, 'NETT_KAP': None, 'TOTAL_PROD': None, 'INSTALLIERT': None, 'STADT_LEVEL': None,
            'EMISSIONEN': None, 'PRICE': None, 'PV_FLAECHE': None, 'POP_DICHTE': None, 'SOLAR_PROD': None, 'SUN1': None
        }, ignore_index=True)

        data2 = data2.sort_values(by='PLZ', ascending=True)
        data2.fillna(method='ffill', inplace=True)
        data2.fillna(method='bfill', inplace=True)

        data3 = data2.loc[data2['LAND_PLZ'] == LAND_PLZ]
        data = data.append(data3, ignore_index=True)

    return data


def correl1(europe, LAND_PLZ, LAND):

        pal=sns.color_palette('viridis', n_colors=10, as_cmap=True)
        sns.scatterplot(x='LNG', y='LAT', s=4.5, data=europe, palette=pal, hue='kwh/kwp_real', cmap='Blues', vmin=2000, vmax=5500)
        location = europe.loc[europe['LAND_PLZ'] == LAND_PLZ]
        sns.scatterplot(x='LNG', y='LAT', s=20, data=location, palette=pal, hue='kwh/kwp_real', cmap='Blues', vmin=2000, vmax=5500)
        plt.show()

        plt.show()
'''
        corr = europe.corr(method='pearson')
        plt.figure(figsize=(20, 6))
        sns.heatmap(corr, annot=True, cmap='Blues')
        plt.title('Correlation matrix')
        plt.show()
        

        sns.set()
        cols = ['LAT', 'LNG', 'kwh/kwp_real']
        sns.pairplot(europe[cols], size=3)
        plt.show()'''


def german_dataset(laender, LAND):

        df = pd.DataFrame()
        LAND1 = LAND
        nuts0 = []
        nuts1 = []
        nuts2 = []
        nuts3 = []
        plz = []
        plz1 = []
        for LAND in laender:
            df1 = pd.read_csv('csv_data/NUTS/pc2020_{}_NUTS-2021_v1.0.csv'.format(LAND), delimiter=';')
            if LAND1 not in laender:
                print('Das Land "{}" ist noch nicht in unserer Datenbank'.format(LAND))
                break
            if LAND != 'UA':
                for x in df1['NUTS3']:
                    nuts3.append(x[1:6])
                    nuts2.append(x[1:5])
                    nuts1.append(x[1:4])
                    nuts0.append(x[1:3])
                for x in df1['CODE']:
                    if LAND == 'UK':
                        x = x.split(' ')
                        x = x[0]
                        plz.append(x[1:])
                    elif LAND == 'NL':
                        plz.append(x[1:5])
                    elif LAND == 'PT':
                        plz.append(x[1:5])
                    else:
                        plz.append(x[1:-1])

            df = df.append(df1)
            plz1.append(plz)
        df['NUTS3'] = nuts3
        df['NUTS2'] = nuts2
        df['NUTS2'] = df['NUTS2'].replace('LU00', 'LU')
        df['NUTS2'] = df['NUTS2'].replace('EE00', 'EE')
        df['NUTS2'] = df['NUTS2'].replace('LV00', 'LV')
        df['NUTS2'] = df['NUTS2'].replace('LT01', 'LT')
        df['NUTS2'] = df['NUTS2'].replace('LT02', 'LT')
        df['NUTS2'] = df['NUTS2'].replace('SI03', 'SI')
        df['NUTS2'] = df['NUTS2'].replace('SI04', 'SI')
        df['NUTS2'] = df['NUTS2'].replace('HR022', 'HR')
        df['NUTS2'] = df['NUTS2'].replace('HR028', 'HR')
        df['NUTS2'] = df['NUTS2'].replace('HR037', 'HR')
        df['NUTS2'] = df['NUTS2'].replace('HR065', 'HR')
        df['NUTS2'] = df['NUTS2'].replace('HR021', 'HR')
        df['NUTS2'] = df['NUTS2'].replace('HR023', 'HR')
        df['NUTS2'] = df['NUTS2'].replace('HR024', 'HR')
        df['NUTS2'] = df['NUTS2'].replace('HR025', 'HR')
        df['NUTS2'] = df['NUTS2'].replace('HR026', 'HR')
        df['NUTS2'] = df['NUTS2'].replace('HR027', 'HR')
        df['NUTS2'] = df['NUTS2'].replace('HR031', 'HR')
        df['NUTS2'] = df['NUTS2'].replace('HR032', 'HR')
        df['NUTS2'] = df['NUTS2'].replace('HR033', 'HR')
        df['NUTS2'] = df['NUTS2'].replace('HR034', 'HR')
        df['NUTS2'] = df['NUTS2'].replace('HR035', 'HR')
        df['NUTS2'] = df['NUTS2'].replace('HR036', 'HR')
        df['NUTS2'] = df['NUTS2'].replace('HR050', 'HR')
        df['NUTS2'] = df['NUTS2'].replace('HR061', 'HR')
        df['NUTS2'] = df['NUTS2'].replace('HR062', 'HR')
        df['NUTS2'] = df['NUTS2'].replace('HR063', 'HR')
        df['NUTS2'] = df['NUTS2'].replace('HR064', 'HR')
        df['NUTS1'] = nuts1
        df['LAND'] = nuts0
        df['PLZ'] = plz
        df['LAND_PLZ'] = df['LAND']+df['PLZ']
        del df['CODE']

        latlong1 = pd.DataFrame()
        for LAND in laender:
            if LAND == 'EL':
                latlong = pd.read_table('csv_data/GEOCOORD/{}.txt'.format(LAND), delimiter=',')
                latlong['LAND'] = 'EL'
                latlong['BUNDESLAND'] = 'EL'
                latlong['PLZ'] = latlong['PLZ'].astype(str)
            else:
                latlong = pd.read_table('csv_data/GEOCOORD/{}.txt'.format(LAND), header=None)
                latlong.rename(columns={0: 'LAND', 1: 'PLZ', 2: 'STADT', 3: 'BUNDESLAND', 8: 'VORWAHL', 9: 'LAT', 10: 'LNG'}, inplace=True)
                del latlong[4], latlong[5], latlong[6], latlong[7], latlong[11], latlong['VORWAHL']
            latlong['PLZ'] = latlong['PLZ'].astype(str)
            latlong['PLZ'] = latlong['PLZ'].drop_duplicates(keep='first')
            latlong['PLZ'] = latlong['PLZ'].astype(str)
            latlong = latlong.loc[latlong['PLZ'] != 'nan']

            if LAND == 'UK':
                latlong['LAND'] = latlong['LAND'].replace(['GB'], 'UK')

            xxx = []
            for i in latlong['PLZ']:
                i = i.replace(' CEDEX', '')
                if LAND == 'UK':
                    i = i
                else:
                    if len(i) == 4:
                        i = '0'+i
                    elif len(i) == 3:
                        i = '00'+i
                    elif len(i) == 2:
                        i = '000'+i
                    elif len(i) == 1:
                        i = '0000'+i
                    else:
                        i = i
                    if LAND == 'BE' or LAND == 'AT' or LAND == 'BG' or LAND == 'CH' or LAND == 'CY' or LAND == 'DK' or LAND == 'HU' or LAND == 'LI' or LAND == 'MK' or LAND == 'NO' or LAND == 'SI' or LAND == 'DK' or LAND == 'NL' or LAND == 'LI':
                        i = i[1:]
                    elif LAND == 'MT' or LAND == 'IE' or LAND == 'IS' or LAND == 'LU':
                        i = i[2:]
                    elif LAND == 'RO':
                        if len(i) == 5:
                            i = '0'+i
                    elif LAND == 'FR':
                        i = i.split(' ')
                        i = i[0]
                    elif LAND == 'PT':
                        i = i[:4]

                xxx.append(i)
            latlong['PLZ'] = xxx

            latlong['LAND_PLZ'] = latlong['LAND'] + latlong['PLZ'].astype(str)
            latlong1 = latlong1.append(latlong, ignore_index=True)
        df = df.drop_duplicates(keep='first')


        del df['LAND'], df['PLZ']
        df1 = pd.merge(latlong1, df, on='LAND_PLZ', how='left')

        country_codes = pd.read_csv('csv_data/energy_data/country_codes.csv')
        del country_codes['alpha-3'], country_codes['iso_3166-2'], country_codes['region'], country_codes['sub-region'], country_codes['intermediate-region'], country_codes['region-code'], country_codes['sub-region-code'], country_codes['intermediate-region-code']
        df = pd.merge(df1, country_codes, on='LAND', how='left')

        net_installed_capacity_of_electric_power_plants_public_solar = pd.read_csv('csv_data/energy_data/UNdata_Export_20221113_215209962.txt', delimiter=';')
        del net_installed_capacity_of_electric_power_plants_public_solar['Commodity - Transaction'], net_installed_capacity_of_electric_power_plants_public_solar['Year'], net_installed_capacity_of_electric_power_plants_public_solar['Quantity Footnotes'], net_installed_capacity_of_electric_power_plants_public_solar['Unit']
        df = pd.merge(df, net_installed_capacity_of_electric_power_plants_public_solar, on='NAME', how='left')

        total_solar = pd.read_csv('csv_data/energy_data/UNdata_Export_20221113_215348121.txt', delimiter=';')
        del total_solar['Commodity - Transaction'], total_solar['Year'], total_solar['Quantity Footnotes'], total_solar['Unit']
        df = pd.merge(df, total_solar, on='NAME', how='left')

        installed = pd.read_csv('csv_data/energy_data/UNdata_Export_20221113_220643718.txt', delimiter=';')
        del installed['Year']
        df = pd.merge(df, installed, on='NAME', how='left')

        cities = pd.read_csv('csv_data/energy_data/city_level_NUTS2.csv')
        del cities['TIME'], cities['GEO_LABEL'], cities['UNIT'], cities['LANDCOVER'], cities['Flag and Footnotes']
        df = pd.merge(df, cities, on='NUTS2', how='left')
        df['STADT_LEVEL'] = df['STADT_LEVEL'].astype(float)

        emissions = pd.read_csv('csv_data/energy_data/emissions.csv')
        emissions = emissions.loc[emissions['TIME_PERIOD'] == 2020]
        del emissions['DATAFLOW'], emissions['LAST UPDATE'], emissions['freq'], emissions['unit'], emissions['OBS_FLAG'], emissions['TIME_PERIOD']
        df = pd.merge(df, emissions, on='LAND', how='left')

        prices = pd.read_csv('csv_data/energy_data/energy_prices1.csv')
        prices = prices.loc[prices['TIME_PERIOD'] == 2021]
        del prices['DATAFLOW'], prices['LAST UPDATE'], prices['freq'], prices['product'], prices['currency'], prices['indic_en'], prices['unit'], prices['TIME_PERIOD'], prices['OBS_FLAG']
        df = pd.merge(df, prices, on='LAND', how='left')

        flaeche = pd.read_csv('csv_data/energy_data/nrg_inf_stcs_1_Data.csv')
        flaeche = flaeche.loc[flaeche['TIME'] == 2020]
        del flaeche['TIME'], flaeche['PLANT_TEC'], flaeche['UNIT'], flaeche['Flag and Footnotes']
        df = pd.merge(df, flaeche, on='NAME', how='left')

        dichte = pd.read_csv('csv_data/energy_data/bahn.csv')
        dichte = dichte.loc[dichte['TIME_PERIOD'] == 2018]
        del dichte['DATAFLOW'], dichte['LAST UPDATE'], dichte['freq'], dichte['unit'], dichte['TIME_PERIOD'], dichte['OBS_FLAG']
        df = pd.merge(df, dichte, on='NUTS2', how='left')
        df['POP_DICHTE'] = df['POP_DICHTE'].astype(float)

        solar_prod = pd.read_csv('csv_data/energy_data/solar_energy_prod.csv')
        solar_prod = solar_prod.loc[solar_prod['TIME'] == 2020]
        del solar_prod['TIME'], solar_prod['SIEC'], solar_prod['UNIT'], solar_prod['Flag and Footnotes'], solar_prod['NRG_BAL']
        df = pd.merge(df, solar_prod, on='NAME', how='left')

        df = df.drop_duplicates(keep='first')
        df['LAND_PLZ'] = df['LAND_PLZ'].astype(str)
        df['LAND_PLZ'] = df['LAND_PLZ'].drop_duplicates(keep='first')
        df['LAND_PLZ'] = df['LAND_PLZ'].astype(str)
        df = df.loc[df['LAND_PLZ'] != 'nan']
        df = df.reset_index()
        del df['index']

        with open('df.textmate', 'w') as file:
            file.write(str(df) + '\n')

        return df


def german_dataset2(german_data, PLZ, LAND_PLZ):

    df1 = pd.read_csv('csv_data/EMHIRES_PVGIS_TSh_CF_n2.csv')
    df1 = df1.reset_index()
    df1.rename(columns={'FR42': 'FRF1','FR61': 'FRI1','FR72': 'FRK1','FR25': 'FRD1','FR26': 'FRC1','FR52': 'FRH0','FR24': 'FRB0','FR21': 'FRF2','FR83': 'FRM0','FR43': 'FRC2','FR23': 'FRD2','FR10': 'FR10','FR81': 'FRJ1','FR63': 'FRI2','FR41': 'FRF3','FR62': 'FRJ2','FR30': 'FRE1','FR51': 'FRG0','FR22': 'FRE2','FR53': 'FRI3','FR82': 'FRL0','FR71': 'FRK2'}, inplace=True)
    df1.rename(columns={'IE02': 'IE05','IE01': 'IE04'}, inplace=True)
    df1.rename(columns={'NO01': 'NO01', 'NO0A': 'NO03', 'NO08': 'NO04', 'NO06': 'NO02', 'NO05': 'NO09'}, inplace=True)
    df1.rename(columns={'SE11 ': 'SE11'}, inplace=True)
    df1.rename(columns={'UKI3UKI4 ': 'UKI3', 'UKI5UKI6 ': 'UKI5'}, inplace=True)

    for i in range(0, 1):
        df2 = pd.read_csv('csv_data/EMHIRESPV_TSh_CF_Country_19862015.csv')
        df2 = df2[len(df2)-len(df1):]
        df2 = df2.reset_index()
        df2['CY'] = df2['CY'].astype(float)*1.25
        df1['SE33'] = df1['SE33'].astype(float)
        df1['LT'] = df2['LT'].astype(float)*1.25
        df1['LU'] = df2['LU'].astype(float)*1.25
        df1['LV'] = df2['LV'].astype(float)*1.25
        df1['EE'] = df2['EE'].astype(float)*1.25
        df1['SI'] = df2['SI'].astype(float)*1.25
        df1['HR'] = df2['HR'].astype(float)*1.25
        x = 5.05
        xx = len(df1)
        df1['TR61'] = 1670 * x / xx
        df1['TRC1'] = 1713 * x / xx
        df1['TR33'] = 1500 * x / xx
        df1['TRA2'] = 1571 * x / xx
        df1['TR83'] = 1572 * x / xx
        df1['TR32'] = 1687 * x / xx
        df1['TR22'] = 1402 * x / xx
        df1['TR41'] = 1439 * x / xx
        df1['TRB1'] = 1587 * x / xx
        df1['TRB2'] = 1600 * x / xx
        df1['TR42'] = 1445 * x / xx
        df1['TRC2'] = 1667 * x / xx
        df1['TR21'] = 1417 * x / xx
        df1['TRA1'] = 1642 * x / xx
        df1['TR90'] = 1150 * x / xx
        df1['TR63'] = 1650 * x / xx
        df1['TR62'] = 1687 * x / xx
        df1['TR10'] = 1385 * x / xx
        df1['TR31'] = 1588 * x / xx
        df1['TR82'] = 1389 * x / xx
        df1['TR72'] = 1614 * x / xx
        df1['TR71'] = 1607 * x / xx
        df1['TR90'] = 1210 * x / xx
        df1['TR51'] = 1532 * x / xx
        df1['TR52'] = 1632 * x / xx
        df1['TRC3'] = 1716 * x / xx
        df1['TR81'] = 1362 * x / xx
        df1['TRA2'] = 1551 * x / xx


    df1['SE11'] = df1['SE11'].astype(float)
    del df1['time_step']
    df1 = df1.loc[df1['year'] > 2012]

    sun1 = []
    for i in range(0, len(german_data['NUTS2'])):
        y = german_data['NUTS2'][i]
        if y in df1.columns:
            xx = sum(df1[str(y)])
            if xx > 0:
                sun1.append(xx)
            else:
                sun1.append(2000)
        else:
            sun1.append(None)

    german_data['SUN1'] = sun1

    german_data = german_data.sort_values(['LAT', 'LNG'], ascending=[True, True])
    german_data.fillna(method='ffill', inplace=True)
    german_data.fillna(method='bfill', inplace=True)
    german_data = german_data.sort_values(['LAND', 'PLZ'], ascending=[True, False])

    your_location = german_data.loc[german_data['LAND_PLZ'] == LAND_PLZ]
    print(your_location)
    nuts2 = your_location['NUTS2'].tolist()
    if len(nuts2) > 0:
        nuts2 = nuts2[0]
    else:
        print('Die PLZ ist ungültig')
    print('Das Solar-Potential wrd für die PV-Anlage berechnet....')
    with open('df1.textmate', 'w') as file:
        file.write(str(german_data) + '\n')
    german_data.to_csv('csv_data/ALL_EU_states2.csv', index=False)

    return german_data


def german_dataset3(german_data, LAND_PLZ):

        german_data1 = german_data.loc[german_data['LAND_PLZ'] == LAND_PLZ]
        print(german_data1)

        return german_data1


def correl2(german_data2):

        corr = german_data2.corr(method='pearson')
        plt.figure(figsize=(20, 6))
        sns.heatmap(corr, annot=True, cmap='Blues')
        plt.title('Correlation matrix')
        plt.show()

        pal=sns.color_palette('viridis', n_colors=10, as_cmap=True)
        sns.scatterplot(x='LNG', y='LAT', s=3, data=german_data2, palette=pal, hue='SUN1', cmap='Blues', vmin=2000, vmax=5500)
        plt.show()

        sns.set()
        cols = ['LAT', 'LNG', 'SUN1']
        sns.pairplot(german_data2[cols], size=3)
        plt.show()


getdata()





