# Compares 2 input names
import json
from app.utils.file_util import load_json_file
from app.utils.name_matcher import NameMatcher
from app.utils.numpy_encoder import NumpyEncoder
from app.utils.preprocess import create_vector_representation, preprocess_single_name
from config import Config

def compare_input_names(name1: str, name2: str):
    model_id = Config.MODEL_ID
    db_data = load_json_file("db.json")
    alias_map = load_json_file("alias_map.json")
    titles = load_json_file("titles.json")
    
    print(f"\nPreprocessing the first name: {name1}")
    processed_name1 = preprocess_single_name(name1, alias_map, titles)
    print(f'processed name1 is {processed_name1}')

    print(f"\nPreprocessing the second name: {name2}")
    processed_name2 = preprocess_single_name(name2, alias_map, titles)
    print(f'processed name2 is {processed_name2}')
    
    processed_names = [processed_name1, processed_name2]
    vector_repr = create_vector_representation(processed_names)
    
    processed_name1['vector_representation'] = vector_repr[processed_name1['original_name']]
    processed_name2['vector_representation'] = vector_repr[processed_name2['original_name']]

    # Initialize matcher with model ID
    matcher = NameMatcher(db_data, [processed_name1], model_id)

    match = matcher.find_matches(processed_name2)

    algorithm_names = matcher.get_phase_2_algorithms()

    # Show results
    result = {
        'algorithm_names': algorithm_names,
        'match': match
    }

    return result



