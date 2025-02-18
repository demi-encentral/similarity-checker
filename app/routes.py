import json
from urllib.parse import urlencode
from flask import Blueprint, make_response, render_template, send_from_directory, request, jsonify
from app.utils import compare_names
from app.utils.numpy_encoder import NumpyEncoder
from config import Config

main_bp = Blueprint('main', __name__)

def custom_jsonify(data, status=200, headers=None):
    response = make_response(json.dumps(data, cls=NumpyEncoder), status)
    response.headers.extend(headers or {})
    response.mimetype = "application/json"
    return response

@main_bp.route('/')
def home():
    form_base_url = Config.FORM_URL 
    return render_template("index.html", form_base_url=form_base_url)

@main_bp.route('/process', methods=['POST'])
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
    }

    # Add algorithm scores and weights dynamically, excluding if not present
    entry_ids = [
        ("entry.2142612982", "entry.1887520687"),
        ("entry.805292406", "entry.1695931903"),
        ("entry.1357732384", "entry.1635473942"),
        ("entry.8241957", "entry.1325311932"),
        ("entry.1823446004", "entry.1821101846"),
        ("entry.1925979563", "entry.2081244785"),
        ("entry.347707063", "entry.296442444"),
        ("entry.226430670", "entry.1658968048"),
        ("entry.1764884276", "entry.1579933235"),
        ("entry.1042142164", "entry.453361351")
    ]

    for i, alg in enumerate(alg_names):
        if i < len(entry_ids):  # Ensure we don't go out of bounds
            score_entry_id, weight_entry_id = entry_ids[i]
            if alg['name'] in result['algorithm_scores']:
                prefilled_data[score_entry_id] = result["algorithm_scores"][alg['name']]
                prefilled_data[weight_entry_id] = alg['version']['weight']

    results["prefilled_data"] = prefilled_data
   
    return jsonify(results)

@main_bp.route('/data/<filename>')
def get_json_file(filename):
    return send_from_directory('static', filename)

@main_bp.route('/feedback')
def feedback():
    form_base_url = Config.FORM_URL 
    return render_template("feedback.html", form_base_url=form_base_url)