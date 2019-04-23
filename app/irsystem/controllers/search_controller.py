from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from app.irsystem.models.search import *

project_name = "KdramaQueen"
net_id = "Adeyemi Oyemade (aao58), Shaima Parveen (sp822), Katie Yang (ky239), Sophie Zhao (sbz24), Chris Elliott (cne27)"
genre_list = pd.read_pickle(os.path.join(os.getcwd(),"app", "irsystem", "models", "Genres.pkl"))
genre_list = list(genre_list)
genre_list.remove('NaN')
titles_list = data['Title']

network_list = ['Channel A','Naver tvcast','Mnet', 'tvN', 'KM' 'Onstyle', 'SBS' 'Netflix', 'KBS', 'MBC', 'DramaX', 'MBN', 'Oksusu',
'UMAX', 'O’live', 'CGV', 'TBS', 'Sohu TV', 'Tooniverse', 'DRAMAcube', 'KBSN', 'E-Channel', 'Fuji TV', 'OCN', 'Yunsae University',
'EBS', 'DramaH','Onstyle', 'CSTV', 'jTBC', 'Viki', "No Preference"]

@irsystem.route('/', methods=['GET', 'POST'])
def search():
	dramas_enjoyed = request.args.get("enjoyed")
	dramas_disliked = request.args.get('disliked')
	preferred_genres = []
	for _ in genre_list:
		if request.args.get('_'):
			preferred_genres.append(_)
	preferred_from  = request.args.get("preferred_from")
	preferred_to = request.args.get("preferred_to")
	preferred_time_frame = []
	if preferred_from and preferred_to:
		preferred_time_frame.append(preferred_from)
		preferred_time_frame.append(preferred_to)
	else:
		preferred_time_frame.append("1938")
		preferred_time_frame.append("2019")

	preferred_networks = []
	for _ in network_list:
		if request.args.get('_'):
			preferred_genres.append(_)
	preferred_actors = request.args.get('preferred_actors')
	num_results = 5


	if not dramas_enjoyed and not preferred_genres:
		output = []
		output_message = ''
		return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, genre=genre_list, titles = titles_list, networks = network_list, output=output)
	else:

		if preferred_genres:
			output_message = "You searched: " + dramas_enjoyed + " with Genre " + preferred_genres
		else:
			output_message = "You searched: " + dramas_enjoyed
		if request.args.get('more-display'):
			num_results += 5
		output = display (dramas_enjoyed, dramas_disliked, preferred_genres, preferred_networks, preferred_actors, preferred_time_frame, num_results)
		return render_template('results.html', name=project_name, netid=net_id, output_message=output_message, genre=genre_list, titles = titles_list, networks = network_list, output=output)
	if request.args.get('new-search'):
		return  render_template('search.html', name=project_name, netid=net_id, output_message=output_message, genre=genre_list, titles = titles_list, networks = network_list, output=output)


			# if request.args.get("name"):
	# 	name = request.args.get("name")
	# 	return render_template('results.html', name=project_name, netid=net_id, output_message=output_message, genre=genre_list, titles = titles_list, networks = network_list, output=output)

# def goback():
# 	btnname = "Make a New Search"
# 	return render_template('search.html')
