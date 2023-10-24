import pandas as pd
from flask import Flask, Blueprint, request, jsonify
import odds_realtime
app = Flask(__name__)
main_blueprint = Blueprint('main', __name__)

# -------------------------------------------- API ------------------------------------------------

@app.route('/api/v2/bookmakers')
def api_soccer_bookmakers():
    df0 = pd.DataFrame()
    df0['bookmaker'] = odds_realtime.soccer_bookmakers()
    return df0.sort_values(by='bookmaker', ascending=True).reset_index(drop=True).transpose().to_dict()

@app.route('/api/v2/allsports')  # gives back all available sports / 1x per year
def all_sports():
    return odds_realtime.getallsports()

@app.route('/api/v2/competitions', methods=["GET"])   # receives sport & gives back comp by sport / 1x per year
def competitions():
    competitions = request.args.get('sport')
    if competitions != None:
        return odds_realtime.getcompetitions(competitions)
    else:
        return jsonify({"message": "Sport is needed to receive competitions."}), 400

@app.route('/api/v2/matches', methods=["GET"])    # receives sport, country, comp & gives back matches with match id, title & start, bookies, status (live?) / 1x per day auto
def matches():
    sport = request.args.get('sport')
    country = request.args.get('country')
    competitions = request.args.get('competition')
    match_urls_key = request.args.get('match_urls')
    if competitions != None and sport != None and country != None and match_urls_key != None:
        return odds_realtime.getmatches(sport, country, competitions, match_urls_key)
    else:
        return jsonify({"message": "Sport, country, competition and match_urls is needed to receive competitions."}), 400

@app.route('/api/v2/odds', methods=["GET"])    # receives sport, country, competition, matchid, bookmakers & gives back odds & co / on demand
def odds():
    sport = request.args.get('sport')
    country = request.args.get('country')
    competition = request.args.get('competition')
    matchid = request.args.get('matchid')
    bookmakers = request.args.get('bookmakers')
    if sport != None and country != None and competition != None and matchid != None and bookmakers != None:
        return odds_realtime.getodds(sport, country, competition, matchid, bookmakers)
    else:
        return jsonify({"message": "Sport, country, competition, matchid and bookmaker(s) is is needed to receive odds."}), 400


@app.route('/api/v2/competitions_bet365', methods=["GET"])   # receives sport & gives back comp by sport / 1x per year
def competitions_bet365():
    competitions = request.args.get('sport')
    if competitions != None:
        df_competitions = pd.read_csv(f'csv_data/{competitions}/leagues.csv')
        df_competitions = df_competitions.loc[df_competitions['bookie'] == 'bet365'][['sports', 'country', 'competition']].drop_duplicates()
        df_competitions = df_competitions.sort_values(by=['sports', 'country', 'competition'], ascending=True).reset_index(drop=True)
        return df_competitions.transpose().fillna(value='').to_dict()
    else:
        return jsonify({"message": "Sport is needed to receive competitions."}), 400

@app.route('/api/v2/matches_bet365', methods=["GET"])    # receives sport, country, comp & gives back matches with match id, title & start, bookies, status (live?) / 1x per day auto
def matches_bet365():
    sport = request.args.get('sport')
    country = request.args.get('country')
    competitions = request.args.get('competition')
    match_urls_key = request.args.get('match_urls')
    if competitions != None and sport != None and country != None and match_urls_key != None:
        odds_realtime.getmatches(sport, country, competitions, match_urls_key)
        same_matches = pd.read_csv(f'csv_data/{sport}/matched4{country}_{competitions}.csv')
        same_matches = same_matches.loc[same_matches['bookie'] == 'bet365'].reset_index(drop=True)
        same_matches = odds_realtime.preprocessing_for_return(same_matches, match_urls_key)
        return same_matches.transpose().fillna(value='').to_dict()
    else:
        return jsonify({"message": "Sport, country, competition and match_urls is needed to receive competitions."}), 400

@app.route('/api/v2/odds_bet365', methods=["GET"])    # receives sport, country, competition, matchid, bookmakers & gives back odds & co / on demand
def odds_bet365():
    sport = request.args.get('sport')
    country = request.args.get('country')
    competition = request.args.get('competition')
    matchid = request.args.get('matchid')
    bookmakers = 'bet365'
    if sport != None and country != None and competition != None and matchid != None and bookmakers != None:
        return odds_realtime.getodds(sport, country, competition, matchid, bookmakers)
    else:
        return jsonify({"message": "Sport, country, competition and matchid is is needed to receive odds."}), 400


@app.route('/api/v2/competitions_pinnacle', methods=["GET"])   # receives sport & gives back comp by sport / 1x per year
def competitions_pinnacle():
    competitions = request.args.get('sport')
    if competitions != None:
        df_competitions = pd.read_csv(f'csv_data/{competitions}/leagues.csv')
        df_competitions = df_competitions.loc[df_competitions['bookie'] == 'pinnacle'][['sports', 'country', 'competition']].drop_duplicates()
        df_competitions = df_competitions.sort_values(by=['sports', 'country', 'competition'], ascending=True).reset_index(drop=True)
        return df_competitions.transpose().fillna(value='').to_dict()
    else:
        return jsonify({"message": "Sport is needed to receive competitions."}), 400

@app.route('/api/v2/matches_pinnacle', methods=["GET"])    # receives sport, country, comp & gives back matches with match id, title & start, bookies, status (live?) / 1x per day auto
def matches_pinnacle():
    sport = request.args.get('sport')
    country = request.args.get('country')
    competitions = request.args.get('competition')
    match_urls_key = request.args.get('match_urls')
    if competitions != None and sport != None and country != None and match_urls_key != None:
        odds_realtime.getmatches(sport, country, competitions, match_urls_key)
        same_matches = pd.read_csv(f'csv_data/{sport}/matched4{country}_{competitions}.csv')
        same_matches = same_matches.loc[same_matches['bookie'] == 'pinnacle'].reset_index(drop=True)
        same_matches = odds_realtime.preprocessing_for_return(same_matches, match_urls_key)
        return same_matches.transpose().fillna(value='').to_dict()
    else:
        return jsonify({"message": "Sport, country, competition and match_urls is needed to receive competitions."}), 400

@app.route('/api/v2/odds_pinnacle', methods=["GET"])    # receives sport, country, competition, matchid, bookmakers & gives back odds & co / on demand
def odds_pinnacle():
    sport = request.args.get('sport')
    country = request.args.get('country')
    competition = request.args.get('competition')
    matchid = request.args.get('matchid')
    bookmakers = 'pinnacle'
    if sport != None and country != None and competition != None and matchid != None and bookmakers != None:
        return odds_realtime.getodds(sport, country, competition, matchid, bookmakers)
    else:
        return jsonify({"message": "Sport, country, competition and matchid is is needed to receive odds."}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

# MATCHES: stake, betathome, cbet, tipwin
# ODDS: daznbet, betclic & cbet 10 odds, betway odds (sportradar)
# ODDS wrong: dafabet handicap
# TODO: basketball odds bet365 & pinnacle, überall 10 mehr odds
