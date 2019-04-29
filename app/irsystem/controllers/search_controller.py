from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from app.irsystem.models.search import *
import re

project_name = "KdramaQueen"
net_id = "Adeyemi Oyemade (aao58), Shaima Parveen (sp822), Katie Yang (ky239), Sophie Zhao (sbz24), Chris Elliott (cne27)"

genre_list = pd.read_pickle(os.path.join(os.getcwd(),"app", "irsystem", "models", "Genres.pkl"))
genre_list = list(genre_list)
genre_list.remove('NaN')

titles_list = pd.read_pickle(os.path.join(os.getcwd(),"app", "irsystem", "models", "Titles.pkl"))

network_list = ['Channel A','Naver tvcast','Mnet', 'tvN', 'KM' 'Onstyle', 'SBS' 'Netflix', 'KBS', 'MBC', 'DramaX', 'MBN', 'Oksusu',
'UMAX', 'O’live', 'CGV', 'TBS', 'Sohu TV', 'Tooniverse', 'DRAMAcube', 'KBSN', 'E-Channel', 'Fuji TV', 'OCN', 'Yunsae University',
'EBS', 'DramaH','Onstyle', 'CSTV', 'jTBC', 'Viki', "No Preference"]



@irsystem.route('/', methods=['GET', 'POST'])

def search():
	dramas_enjoyed = request.args.get("enjoyed")
	dramas_disliked = request.args.get('disliked')
	preferred_genres =request.args.get("preferred_genres")
	preferred_from  = request.args.get("preferred_from")
	preferred_to = request.args.get("preferred_to")
	preferred_time_frame = []
	if preferred_from and preferred_to:
		preferred_time_frame.append(preferred_from)
		preferred_time_frame.append(preferred_to)
	else:
		preferred_time_frame.append("1958")
		preferred_time_frame.append("2019")

	preferred_actors = request.args.get('preferred_actors')
	clicked_img = request.args.get("img-click")
	num_results = 21
	

	if not dramas_enjoyed and not dramas_disliked and not preferred_genres and not preferred_actors:
		output = []
		output_message = ''
		return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, genre=genre_list, titles = titles_list, clicked_img = clicked_img, networks = network_list, output=output)
	else:
		if type(dramas_enjoyed) == None and type(dramas_disliked) == None and type(preferred_genres) == None and type(preferred_actors) == None:
			output = display('','','','', preferred_time_frame, num_results)
		else:
			output_message = ''
			if (dramas_enjoyed):
				output_message += "Similar shows to {}.".format(dramas_enjoyed)
			if (dramas_disliked): 
				output_message += "Shows different from {}.".format(dramas_disliked)
			if (preferred_genres):	
				output_message += "Should contain the {} genre(s)".format(preferred_genres)
			if (preferred_actors):
				output_message +=" and these actor(s): {}.".format(preferred_actors)
			output_message += "Preferably between {} to {} years.".format(preferred_time_frame[0], preferred_time_frame[1])
			output = display (dramas_enjoyed, dramas_disliked, preferred_genres, preferred_actors, preferred_time_frame, num_results)
		return render_template('results.html', name=project_name, netid=net_id, output_message=output_message, genre=genre_list, titles = titles_list, clicked_img = clicked_img, networks = network_list, output=output)
	if request.args.get('new-search'):
		return  render_template('search.html', name=project_name, netid=net_id, output_message=output_message, genre=genre_list, titles = titles_list,  clicked_img = clicked_img, networks = network_list, output=output)
	# if request.args.get('more-display')
	# 	return render_template('results.html', name=project_name, netid=net_id, output_message=output_message, genre=genre_list, titles = titles_list, networks = network_list, output=output)

