import re
from typing import List, Dict, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from unidecode import unidecode

vectorizer = TfidfVectorizer(
    analyzer='char_wb',  # Use character n-grams with word boundaries
    ngram_range=(2, 4),  # A broader range to capture various name components
    min_df=1,  # Include all terms, even those that appear only once, since you're dealing with few documents
    max_df=1.0,  # No upper limit on term frequency since common terms might be relevant
    lowercase=True,  # Convert all characters to lowercase for consistency
    strip_accents='unicode'  # Normalize accented characters which could appear in names
)

def normalize_text(text: str) -> str:
    """
    Normalize text by converting to lowercase, removing accents and special characters.
    """
    text = text.lower()
    text = unidecode(text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = ' '.join(text.split())
    return text

def generate_ngrams(text: str, n: int) -> Set[str]:
    """
    Generate character n-grams from text.
    """
    padded = f"{'_' * (n-1)}{text}{'_' * (n-1)}"
    ngrams = set()
    for i in range(len(padded) - n + 1):
        ngrams.add(padded[i:i+n])
    return ngrams

def compute_ngrams(text: str) -> Dict[str, Set[str]]:
    """
    Compute 2-gram, 3-gram, and 4-gram representations.
    """
    return {
        'bigrams': generate_ngrams(text, 2),
        'trigrams': generate_ngrams(text, 3),
        'quadgrams': generate_ngrams(text, 4)
    }


def create_vector_representation(names: List[Dict]) -> Dict[str, List[float]]:
    """
    Create TF-IDF vector representations for all names.
    """
    print("Creating vector representations...")
    normalized_names = [name['normalized_name'] for name in names]
    tfidf_matrix = vectorizer.fit_transform(normalized_names)

    vector_dict = {}
    for i, name in enumerate(names):
        vector = tfidf_matrix[i].toarray()[0]
        vector_dict[name['original_name']] = vector.tolist()

    print(f"Created vectors with {len(vectorizer.get_feature_names_out())} features")
    return vector_dict

def tokenize_name(name: str) -> List[str]:
    """Split name into tokens and sort them."""
    return sorted(name.split())

def remove_titles(name: str, titles: List[str]) -> str:
    """Remove common titles from names."""
    name_lower = name.lower()
    for title in titles:
        pattern = f"\\b{title.lower()}\\b"
        name_lower = re.sub(pattern, '', name_lower)
    return ' '.join(name_lower.split())

def replace_aliases(tokens: List[str], alias_dict: Dict[str, str]) -> List[str]:
    """Replace nicknames/abbreviations with their full forms using alias dictionary."""
    return [alias_dict.get(token, token) for token in tokens]


def preprocess_single_name(
    name: str,
    alias_dict: Dict[str, str],
    titles: List[str] 
) -> Dict:
    """
    Preprocess a single name using the same pipeline as batch processing.

    Args:
        input_name (str): The name to preprocess.
        alias_dict (Dict[str, str]): Dictionary of aliases for replacing abbreviations or nicknames.
        titles (List[str]): List of titles to remove from the name.

    Returns:
        Dict: Processed name with original name, normalized name, tokens, and n-grams and its vector representation.
    """
    print(f"\nProcessing single name: {name}")

    # Remove titles
    name_no_titles = remove_titles(name, titles)
    print(f"After title removal: {name_no_titles}")

    # Normalize name
    normalized_name = normalize_text(name_no_titles)
    print(f"After normalization: {normalized_name}")

    # Tokenize name
    tokens = tokenize_name(normalized_name)
    print(f"After tokenization: {tokens}")

    # Replace aliases
    processed_tokens = replace_aliases(tokens, alias_dict)
    print(f"After alias replacement: {processed_tokens}")

    # Compute n-grams
    ngrams = compute_ngrams(normalized_name)
    print(f"Generated n-grams: {len(ngrams['bigrams'])} bigrams, {len(ngrams['trigrams'])} trigrams, {len(ngrams['quadgrams'])} quadgrams")


    # Assemble the result
    result = {
        "original_name": name,
        "normalized_name": ' '.join(processed_tokens),
        "tokens": processed_tokens,
        "ngrams": {
            "bigrams": list(ngrams['bigrams']),
            "trigrams": list(ngrams['trigrams']),
            "quadgrams": list(ngrams['quadgrams']),
        }
    }

    return result
