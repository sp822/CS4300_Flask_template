from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from app.irsystem.models.search import *

project_name = "KdramaQueen"
net_id = "sp822, sbz24, ky239, cne27, aao58"

@irsystem.route('/', methods=['GET'])
def search():
	enjoyed = request.args.get("enjoyed")
	disliked = request.args.get('disliked')
	prefered_genres = request.args.get('prefered_genres')
	prefered_networks = request.args.get('prefered_networks')

	
	if not enjoyed and not prefered_genres:
		output = []
		output_message = ''
	else:
		if prefered_genres:
			output_message = "You searched: " + enjoyed + " with Genre " + prefered_genres
		else: 
			output_message = "You searched: " + enjoyed
		output = display(num_movies, movie_sims_cos, data, movie_index_to_name, movie_name_to_index, enjoyed, disliked, prefered_genres, prefered_networks, 5)

	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, output=output)
