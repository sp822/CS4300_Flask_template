from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from app.irsystem.models.search import *

project_name = "KdramaQueen"
net_id = "sp822, sbz24, ky239, cne27, aao58"

@irsystem.route('/', methods=['GET'])
def search():
	dramas_enjoyed = request.args.get("enjoyed")
	dramas_disliked = request.args.get('disliked')
	preferred_genres = request.args.get('prefered_genres')
	preferred_networks = request.args.get('prefered_networks')
	preferred_actors = request.args.get('preferred_actors')
	preferred_time_frame = request.args.get('preferred_time_frame')

	
	if not dramas_enjoyed and not preferred_genres:
		output = []
		output_message = ''
		return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, output=output)
	else:
		if preferred_genres:
			output_message = "You searched: " + dramas_enjoyed + " with Genre " + preferred_genres
		else: 
			output_message = "You searched: " + dramas_enjoyed
		output = display (dramas_enjoyed, dramas_disliked, preferred_genres, preferred_networks, preferred_actors, preferred_time_frame, 5)

		return render_template('results.html', name=project_name, netid=net_id, output_message=output_message, output=output)
	if request.args.get('new-search'):
		return  render_template('search.html', name=project_name, netid=net_id, output_message=output_message, output=output)

# def goback():
# 	btnname = "Make a New Search"
# 	return render_template('search.html')