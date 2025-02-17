from urllib.parse import urlencode
from flask import Blueprint, render_template, send_from_directory, request, jsonify
from app.utils import compare_names
from config import Config

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def home():
    return render_template("index.html")

@main_bp.route('/process', methods= ['POST'])
def process():
    data = request.get_json()

    # Extract name1 and name2 from the JSON data
    name1 = data.get('name1')
    name2 = data.get('name2')
    results = compare_names.compare_input_names(name1, name2)
    result = results["match"]
    alg_names = results["algorithm_names"]
   
    prefilled_data = {
    "usp": "pp_url", 
    "entry.2072591192": result["candidate_name"],
    "entry.872788899": result["query_name"],
    "entry.828234223": result["confidence"],
    "entry.749418275": result["final_score"],
    "entry.2142612982": result["algorithm_scores"][alg_names[0]["name"]],
    "entry.805292406": result["algorithm_scores"][alg_names[1]["name"]],
    "entry.1357732384": result["algorithm_scores"][alg_names[2]["name"]],
    "entry.8241957": result["algorithm_scores"][alg_names[3]["name"]],
    "entry.1823446004": result["algorithm_scores"][alg_names[4]["name"]],
    "entry.1925979563": result["algorithm_scores"][alg_names[5]["name"]],
    "entry.347707063": result["algorithm_scores"][alg_names[6]["name"]],
    "entry.226430670": result["algorithm_scores"][alg_names[7]["name"]],
    "entry.1764884276": result["algorithm_scores"][alg_names[8]["name"]]
    }
    # prefilled_url = f"{form_base_url}?{urlencode(prefilled_data)}"
    # results["google_form_url"] = prefilled_url
    results["prefilled_data"] = prefilled_data

    return jsonify(results)


@main_bp.route('/data/<filename>')
def get_json_file(filename):
    return send_from_directory('static', filename)

@main_bp.route('/feedback')
def feedback():
    form_base_url = Config.FORM_URL 
    return render_template("feedback.html", form_base_url=form_base_url)