from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from app.irsystem.models.search import *

project_name = "KdramaQueen"
net_id = "sp822, sbz24, ky239, cne27, aao58"

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('enjoyed')
	if not query:
		output = []
		output_message = ''
	else:
		output_message = "Your search: " + query
		output = range(5)
		output = best_match(num_movies, movie_sims_cos, data, movie_index_to_name, movie_name_to_index, ["Doctor Stranger","Doctors","Emergency Couple"], [], ["Romance","Medical"], ["fun"], int(query))
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, output=output)
