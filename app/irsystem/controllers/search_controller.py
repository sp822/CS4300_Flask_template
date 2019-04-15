from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from app.irsystem.models.search import *

project_name = "KdramaQueen"
net_id = "sp822, sbz24, ky239, cne27, aao58"

@irsystem.route('/', methods=['GET'])
def search():
	enjoyed = request.args.get('enjoyed')
	if not enjoyed:
		output = []
		output_message = ''
	else:
		output_message = "Your search: " + enjoyed
		output = best_match(num_movies, movie_sims_cos, data, movie_index_to_name, movie_name_to_index, enjoyed, [], ["Romance","Medical"], ["fun"], 10)
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, output=output)
