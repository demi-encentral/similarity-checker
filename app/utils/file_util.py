import os
import json
from typing import Dict, List
from flask import current_app
from logging_config import logger

def ensure_static_path(filepath: str) -> str:
    """Ensure the given filepath is within the static directory."""
    return os.path.join(current_app.static_folder, filepath)

def load_json_file(filepath: str) -> Dict:
    """Load and return contents of a JSON file."""
    full_path = os.path.join(current_app.static_folder, filepath)
    logger.debug(f"Loading {full_path}...")
    with open(full_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json_file(data: Dict, filepath: str) -> None:
    """Save data to a JSON file."""
    full_path = os.path.join(current_app.static_folder, filepath) 
    logger.debug(f"Saving processed data to {full_path}...")
    with open(full_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def save_processed_data(processed_names: List[Dict], filepath: str) -> None:
    """Save processed data with special handling for sets."""
    # Convert sets to lists only during saving
    serializable_data = []
    for name_entry in processed_names:
        serializable_entry = name_entry.copy()
        serializable_entry['ngrams'] = {
            'bigrams': list(name_entry['ngrams']['bigrams']),
            'trigrams': list(name_entry['ngrams']['trigrams']),
            'quadgrams': list(name_entry['ngrams']['quadgrams'])
        }
        serializable_data.append(serializable_entry)

    # Ensure filepath is within the static directory
    full_path = os.path.join(current_app.static_folder, filepath)

    # Write data to the file (overwrites if exists, creates if not)
    logger.debug(f"Saving processed data to {full_path}...")
    with open(full_path, 'w', encoding='utf-8') as file:
        json.dump({'processed_names': serializable_data}, file, indent=2)
