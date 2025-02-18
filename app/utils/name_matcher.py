import json
from typing import List, Dict, Tuple, Set
import numpy as np
from collections import defaultdict
from jarowinkler import jarowinkler_similarity as jaro_winkler
from rapidfuzz import fuzz
from logging_config import logger

class NameMatcher:
    def __init__(self, config: Dict, preprocessed_names: List[Dict], model_version_id: str):
        self.config = config
        self.names = preprocessed_names
        self.model_version_id = model_version_id
        self.setup_algorithms()

    def setup_algorithms(self):
        """Organize algorithms into phases based on contributes_to_score flag for the specific model version"""
        self.phase1_algorithms = []
        self.phase2_algorithms = []
        # Find the correct model version
        model_version = next((mv for mv in self.config['model_versions'] if mv['version_id'] == self.model_version_id), None)
        if not model_version:
            raise ValueError(f"Model version {self.model_version_id} not found.")
        self.max_number_of_matches = model_version['max_number_of_matches']
        for alg_config in model_version['config']:
            algorithm_info = {
                'name': alg_config['algorithm'],
                'version': alg_config,
                'execution_order': alg_config['execution_order']
            }
            if not alg_config['contributes_to_score']:
                self.phase1_algorithms.append(algorithm_info)
            else:
                self.phase2_algorithms.append(algorithm_info)
        # Sort algorithms by execution order within each phase
        self.phase1_algorithms.sort(key=lambda x: x['execution_order'])
        self.phase2_algorithms.sort(key=lambda x: x['execution_order'])


    def hash_lookup(self, query: Dict, candidate: Dict) -> float:
        """Fast hash-based lookup using tokens and bigrams"""
        # Check token overlap
        query_tokens = set(query['tokens'])
        candidate_tokens = set(candidate['tokens'])
        token_overlap = len(query_tokens.intersection(candidate_tokens))

        if token_overlap == 0:
            logger.debug("Hash lookup: No token overlap")
            return 0

        # Check bigram overlap
        query_bigrams = set(query['ngrams']['bigrams'])
        candidate_bigrams = set(candidate['ngrams']['bigrams'])
        bigram_overlap = len(query_bigrams.intersection(candidate_bigrams))
        bigram_similarity = bigram_overlap / max(len(query_bigrams), len(candidate_bigrams))

        logger.debug(f"Hash lookup - Token overlap: {token_overlap}, Bigram similarity: {bigram_similarity:.3f}")
        return bigram_similarity
    
  
    def is_initial(self, token: str) -> bool:
        return len(token) == 1 and token.isalpha()

    
    def subset_name_similarity(self, query: Dict, candidate: Dict) -> float:
        """
        Calculate similarity score for names where one might be a subset of the other,
        considering initials and penalizing typos more leniently while also penalizing 
        differing initials. This method checks both directions for subset relationship 
        with improved initial matching and typo tolerance, regardless of token order.

        :param query: Dictionary containing tokens and normalized name of the query name
        :param candidate: Dictionary containing tokens and normalized name of the candidate name
        :return: Similarity score between 0 and 1
        """

        def check_initial_mismatch(initial: str, full_tokens: set) -> bool:
            """
            Check if an initial explicitly mismatches any full token's first letter.
            Returns True if we find a token that definitely doesn't match this initial.
            """
            for token in full_tokens:
                if len(token) > 1 and token.lower().startswith(initial.lower()):
                    return False  # Found a matching token
            return True  # No matching token found

        def calculate_subset_score(q_tokens: set, c_tokens: set) -> float:
            expanded_query_tokens = set()
            initials_to_check = set()
            
            # First pass: separate initials and non-initials
            for token in q_tokens:
                if self.is_initial(token):
                    initials_to_check.add(token)
                else:
                    expanded_query_tokens.add(token)
            
            # Check for explicit initial mismatches
            for initial in initials_to_check:
                if check_initial_mismatch(initial, c_tokens):
                    return 0.75  # Strong penalty for definite initial mismatch
                
                # Add any matching full tokens
                for cand_token in c_tokens:
                    if len(cand_token) > 1 and cand_token.lower().startswith(initial.lower()):
                        expanded_query_tokens.add(cand_token)
                        break
            
            # Check if remaining tokens form a subset
            if expanded_query_tokens.issubset(c_tokens):
                if not initials_to_check:
                    return 1.0  # Perfect match without initials
                return 0.9  # Good match with matching initials
            
            # Calculate fuzzy match score for non-subset cases
            return fuzz.partial_ratio(query['normalized_name'], 
                                    candidate['normalized_name']) / 100.0

        query_tokens, candidate_tokens = set(query['tokens']), set(candidate['tokens'])
        score1 = calculate_subset_score(query_tokens, candidate_tokens)
        score2 = calculate_subset_score(candidate_tokens, query_tokens)

        # Take the higher score unless one direction shows a definite initial mismatch
        if score1 == 0.75 or score2 == 0.75:
            return 0.75
        return max(score1, score2)


    def is_initial(self, token: str) -> bool:
        return len(token) == 1 and token.isalpha() 
    
    def set_intersections(self, query: Dict, candidate: Dict) -> float:
        """Set intersection using n-grams"""
        # Use all n-gram types for more comprehensive comparison
        similarities = []

        for gram_type in ['bigrams', 'trigrams', 'quadgrams']:
            query_grams = set(query['ngrams'][gram_type])
            candidate_grams = set(candidate['ngrams'][gram_type])

            intersection = len(query_grams.intersection(candidate_grams))
            union = len(query_grams.union(candidate_grams))

            if union > 0:
                similarities.append(intersection / union)

        avg_similarity = sum(similarities) / len(similarities)
        logger.debug(f"Set intersection similarity: {avg_similarity:.3f}")
        return avg_similarity

    def cosine_distance(self, query: Dict, candidate: Dict) -> float:
        """Cosine similarity using vector representations"""
        query_vector = np.array(query['vector_representation'])
        candidate_vector = np.array(candidate['vector_representation'])

        # Calculate cosine similarity
        dot_product = np.dot(query_vector, candidate_vector)
        norm_product = np.linalg.norm(query_vector) * np.linalg.norm(candidate_vector)

        if norm_product == 0:
            return 0

        similarity = dot_product / norm_product
        logger.debug(f"Cosine similarity: {similarity:.3f}")
        return similarity

    def exact_match(self, query: Dict, candidate: Dict) -> float:
        """Check for exact token match"""
        is_match = sorted(query['tokens']) == sorted(candidate['tokens'])
        score = 1.0 if is_match else 0.0
        logger.debug(f"Exact match score: {score}")
        return score

    def normalized_token_overlap(self, query: Dict, candidate: Dict) -> float:
        """Calculate normalized token overlap with n-gram support"""
        # Token-level overlap
        query_tokens = set(query['tokens'])
        candidate_tokens = set(candidate['tokens'])
        token_overlap = len(query_tokens.intersection(candidate_tokens))
        token_score = token_overlap / max(len(query_tokens), len(candidate_tokens))

        # N-gram overlap (using bigrams and trigrams)
        gram_scores = []
        for gram_type in ['bigrams', 'trigrams']:
            query_grams = set(query['ngrams'][gram_type])
            candidate_grams = set(candidate['ngrams'][gram_type])
            overlap = len(query_grams.intersection(candidate_grams))
            gram_scores.append(overlap / max(len(query_grams), len(candidate_grams)))

        # Combine scores (giving more weight to token-level overlap)
        final_score = 0.6 * token_score + 0.4 * (sum(gram_scores) / len(gram_scores))
        logger.debug(f"Token overlap score: {final_score:.3f}")
        return final_score

    def token_sort_ratio(self, query: Dict, candidate: Dict) -> float:
        """Calculate token sort ratio using normalized names"""
        score = fuzz.token_sort_ratio(query['normalized_name'],
                                    candidate['normalized_name']) / 100.0
        logger.debug(f"Token sort ratio: {score:.3f}")
        return score

    def jaccard_similarity(self, query: Dict, candidate: Dict) -> float:
        """Calculate Jaccard similarity using tokens and n-grams"""
        # Calculate for tokens
        query_tokens = set(query['tokens'])
        candidate_tokens = set(candidate['tokens'])
        token_jaccard = len(query_tokens.intersection(candidate_tokens)) / \
                       len(query_tokens.union(candidate_tokens))

        # Calculate for n-grams
        gram_scores = []
        for gram_type in ['bigrams', 'trigrams']:
            query_grams = set(query['ngrams'][gram_type])
            candidate_grams = set(candidate['ngrams'][gram_type])
            intersection = len(query_grams.intersection(candidate_grams))
            union = len(query_grams.union(candidate_grams))
            gram_scores.append(intersection / union)

        # Combine scores
        final_score = 0.5 * token_jaccard + 0.5 * (sum(gram_scores) / len(gram_scores))
        logger.debug(f"Jaccard similarity: {final_score:.3f}")
        return final_score

    def jaro_winkler_similarity(self, query: Dict, candidate: Dict) -> float:
        """Calculate Jaro-Winkler similarity using normalized names"""
        score = jaro_winkler(query['normalized_name'],
                                candidate['normalized_name'])
        logger.debug(f"Jaro-Winkler similarity: {score:.3f}")
        return score

    def levenshtein_distance(self, query: Dict, candidate: Dict) -> float:
        """Calculate normalized Levenshtein similarity"""
        score = fuzz.ratio(query['normalized_name'],
                         candidate['normalized_name']) / 100.0
        logger.debug(f"Levenshtein similarity: {score:.3f}")
        return score


    def apply_phase1_filters(self, query: Dict, candidate: Dict) -> bool:
        for algorithm in self.phase1_algorithms:
            min_score = algorithm['version']['min_score']
            func = getattr(self, algorithm['name'].lower().replace(' ', '_'))
            if func(query, candidate) < min_score:
                logger.debug(f"Failed {algorithm['name']} filter")
                return False
        logger.debug("Passed all Phase 1 filters")
        return True

    def calculate_phase2_scores(self, query: Dict, candidate: Dict) -> Tuple[float, Dict[str, float]]:
        scores = {}
        total_score = 0.0
        total_weight = 0.0
        for algorithm in self.phase2_algorithms:
            func = getattr(self, algorithm['name'].lower().replace(' ', '_'))
            weight = algorithm['version']['weight']
            threshold = algorithm['version']['threshold']
            score = func(query, candidate)
            scores[algorithm['name']] = score
            if score >= threshold:
                logger.debug(f"Threshold met by {algorithm['name']}, terminating scoring with score {score:.3f}")
                return score, scores  # Use this score directly instead of weighted average
            if algorithm['version'].get('terminates_on_exact_match', False) and score == 1.0:
                logger.debug(f"Exact match found in {algorithm['name']}, terminating scoring")
                return 1.0, scores
            total_score += score * weight
            total_weight += weight
        final_score = total_score / total_weight if total_weight > 0 else 0.0
        logger.debug(f"No algorithm met threshold, using weighted average: {final_score:.3f}")
        return final_score, scores

    def find_matches(self, query_name: Dict) -> List[Dict]:
        matches = []
        logger.debug(f"\nProcessing query name: {query_name['original_name']}")
        logger.debug(f"\nMax number of matches: {self.max_number_of_matches}")
        
        # Step 1: Check for exact match
        for candidate in self.names:
            if candidate['original_name'] == query_name['original_name']:
                match_result = {
                    'candidate_name': candidate['original_name'],
                    'final_score': 1.0,
                    'algorithm_scores': {'Exactly equals': 1.0},
                    'confidence': 'RECOMMENDED'
                }
                matches.append(match_result)
                self.names.remove(candidate)  # Remove exact match from further processing
                if len(matches) >= self.max_number_of_matches:
                    logger.debug(f"Reached maximum number of matches ({self.max_number_of_matches}). Stopping search.")
                    return matches

        # Step 2: Phase 1 filtering
        phase1_candidates = []
        for candidate in self.names:
            if self.apply_phase1_filters(query_name, candidate):
                phase1_candidates.append(candidate)

        # Step 3 & 4: Phase 2 scoring
        for candidate in phase1_candidates:
            logger.debug(f"\nEvaluating candidate: {candidate['original_name']}")
            final_score, algorithm_scores = self.calculate_phase2_scores(query_name, candidate)
            logger.debug(f"Final score: {final_score:.3f}")

            confidence = ''
            if final_score == 1.0:
                confidence = 'EXACT MATCH'
            elif final_score >= 0.80:
                confidence = 'HIGHLY RECOMMENDED'
            elif final_score >= 0.70:
                confidence = 'MODERATELY RECOMMENDED'
            else:    
                confidence = 'NO MATCH'

            match_result = {
                'candidate_name': candidate['original_name'],
                'query_name': query_name['original_name'],
                'final_score': final_score,
                'algorithm_scores': algorithm_scores,
                'confidence': confidence
            }

            matches.append(match_result)
            if len(matches) >= self.max_number_of_matches:
                logger.debug(f"Reached maximum number of matches ({self.max_number_of_matches}). Stopping search.")
                break

        # Return only one match since max_number_of_matches is set to 1 for mv003
        return matches[0] if matches else None
    
    def get_phase_1_algorithms(self) -> List[str]:
        return self.phase1_algorithms
    
    def get_phase_2_algorithms(self) -> List[str]:
        return self.phase2_algorithms

   


def print_matching_results(results: List[Dict]):
    """print formatted matching results"""
    logger.debug("\n=== DETAILED MATCHING RESULTS ===\n")

    for result in results:
        logger.debug(f"Query Name: {result['query_name']}")
        logger.debug("Matches:")

        for idx, match in enumerate(result['matches'], 1):
            logger.debug(f"\n  {idx}. Candidate: {match['candidate_name']}")
            logger.debug(f"     Final Score: {match['final_score']:.3f}")
            logger.debug(f"     Confidence: {match['confidence'].upper()}")
            logger.debug("\n     Algorithm Scores:")

            # logger.debug individual algorithm scores sorted by score value
            sorted_scores = sorted(
                match['algorithm_scores'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for alg_name, score in sorted_scores:
                logger.debug(f"       - {alg_name}: {score:.3f}")

        logger.debug("\n" + "="*50 + "\n")

    # logger.debug summary statistics
    total_matches = sum(len(result['matches']) for result in results)
    avg_matches = total_matches / len(results) if results else 0

    logger.debug("Summary Statistics:")
    logger.debug(f"Total names with matches: {len(results)}")
    logger.debug(f"Total matches found: {total_matches}")
    logger.debug(f"Average matches per name: {avg_matches:.2f}")

    # logger.debug confidence level breakdown
    confidence_counts = defaultdict(int)
    for result in results:
        for match in result['matches']:
            confidence_counts[match['confidence']] += 1

    logger.debug("\nConfidence Level Breakdown:")
    for confidence in ['EXACT_MATCH', 'RECOMMENDED', 'NO_MATCH']:
        count = confidence_counts[confidence]
        percentage = (count / total_matches * 100) if total_matches > 0 else 0
        logger.debug(f"{confidence.upper()}: {count} ({percentage:.1f}%)")