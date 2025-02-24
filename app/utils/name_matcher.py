from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Set
import numpy as np
from collections import defaultdict
from jarowinkler import jarowinkler_similarity as jaro_winkler
from rapidfuzz import fuzz

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
        print(f"Setting up Name matcher for model: {self.model_version_id} with {len(self.phase1_algorithms)} algorithms in phase 1 and {len(self.phase2_algorithms)} algorithms in phase 2")


    def hash_lookup(self, query: Dict, candidate: Dict) -> float:
        """Fast hash-based lookup using tokens and bigrams"""
        # Check token overlap
        query_tokens = set(query['tokens'])
        candidate_tokens = set(candidate['tokens'])
        token_overlap = len(query_tokens.intersection(candidate_tokens))

        if token_overlap == 0:
            print("Hash lookup: No token overlap")
            return 0

        # Check bigram overlap
        query_bigrams = set(query['ngrams']['bigrams'])
        candidate_bigrams = set(candidate['ngrams']['bigrams'])
        bigram_overlap = len(query_bigrams.intersection(candidate_bigrams))
        bigram_similarity = bigram_overlap / max(len(query_bigrams), len(candidate_bigrams))

        print(f"Hash lookup - Token overlap: {token_overlap}, Bigram similarity: {bigram_similarity:.3f}")
        return bigram_similarity

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
        print(f"Set intersection similarity: {avg_similarity:.3f}")
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
        print(f"Cosine similarity: {similarity:.3f}")
        return similarity

    def exact_match(self, query: Dict, candidate: Dict) -> float:
        """Check for exact token match"""
        is_match = sorted(query['tokens']) == sorted(candidate['tokens'])
        score = 1.0 if is_match else 0.0
        print(f"Exact match score: {score}")
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
        print(f"Token overlap score: {final_score:.3f}")
        return final_score

    def token_sort_ratio(self, query: Dict, candidate: Dict) -> float:
        """Calculate token sort ratio using normalized names"""
        score = fuzz.token_sort_ratio(query['normalized_name'],
                                    candidate['normalized_name']) / 100.0
        print(f"Token sort ratio: {score:.3f}")
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
        print(f"Jaccard similarity: {final_score:.3f}")
        return final_score

    def jaro_winkler_similarity(self, query: Dict, candidate: Dict) -> float:
        """Calculate Jaro-Winkler similarity using normalized names"""
        score = jaro_winkler(query['normalized_name'],
                                candidate['normalized_name'])
        print(f"Jaro-Winkler similarity: {score:.3f}")
        return score

    def levenshtein_distance(self, query: Dict, candidate: Dict) -> float:
        """Calculate normalized Levenshtein similarity"""
        score = fuzz.ratio(query['normalized_name'],
                         candidate['normalized_name']) / 100.0
        print(f"Levenshtein similarity: {score:.3f}")
        return score
    
    def edit_distance(self, s1: str, s2: str) -> int:
        """Compute the Levenshtein edit distance between s1 and s2."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,     # deletion
                    dp[i][j - 1] + 1,     # insertion
                    dp[i - 1][j - 1] + cost  # substitution
                )
        return dp[m][n]
    
    def subset_name_similarity(self, query: dict, candidate: dict, penalty_per_edit=0.1, unmatched_penalty=0.01, start_letter_penalty=0.25, initial_match_score=0.9, threshold=0.7) -> float:
        """
        Calculate an improved similarity score between two names using exact Levenshtein distance.
        Matches tokens greedily, penalizes typos, accounts for unmatched tokens, applies a penalty
        for mismatched starting letters in non-exact matches, and handles initials intelligently.

        Args:
            query (dict): Dictionary with 'tokens' key (list of strings) for the query name.
            candidate (dict): Dictionary with 'tokens' key (list of strings) for the candidate name.
            penalty_per_edit (float): Penalty multiplier per edit distance (default 0.1).
            unmatched_penalty (float): Penalty per unmatched token in the larger set (default 0.01).
            start_letter_penalty (float): Penalty for mismatched starting letters in non-exact matches (default 0.05).
            initial_match_score (float): Similarity score for a full name matching an initial (default 0.9).
            threshold (float): Minimum similarity for non-initial matches to override initial matching (default 0.7).

        Returns:
            float: Similarity score between 0.0 and 1.0.
        """
        query_tokens = query.get('tokens', [])
        candidate_tokens = candidate.get('tokens', [])
        
        print("Query tokens:", query_tokens)
        print("Candidate tokens:", candidate_tokens)
        
        # Handle edge cases
        if not query_tokens and not candidate_tokens:
            print("Both names are empty. Returning 1.0")
            return 1.0
        if not query_tokens or not candidate_tokens:
            print("One of the names is empty. Returning 0.0")
            return 0.0

        # Determine smaller and larger token sets
        if len(query_tokens) <= len(candidate_tokens):
            smaller_tokens = query_tokens
            larger_tokens = candidate_tokens.copy()
        else:
            smaller_tokens = candidate_tokens
            larger_tokens = query_tokens.copy()
        
        print("Using smaller token set:", smaller_tokens)
        
        # Helper function to check if a token is an initial
        def is_initial(token):
            return len(token) == 1 and token.isalpha()

        # Match tokens greedily
        total_similarity = 0.0
        matched_indices = []
        all_exact_matches = True  # Track if all matches are exact
        max_non_initial_similarity = 0.0  # Track highest non-initial match
        
        for token in smaller_tokens:
            best_similarity = 0.0
            best_index = None
            best_distance = float('inf')
            best_candidate = None
            best_is_initial_match = False  # Track if best match is an initial match
            print(f"Matching token '{token}' against candidates: {larger_tokens}")
            
            for i, cand_token in enumerate(larger_tokens):
                if i in matched_indices:
                    continue
                # Check for initial matching
                if is_initial(token) and not is_initial(cand_token):
                    similarity = initial_match_score if cand_token[0].lower() == token.lower() else 0.0
                    is_initial_match = True
                elif not is_initial(token) and is_initial(cand_token):
                    similarity = initial_match_score if token[0].lower() == cand_token.lower() else 0.0
                    is_initial_match = True
                else:
                    # Standard edit distance-based similarity
                    distance = self.edit_distance(token.lower(), cand_token.lower())
                    if distance == 0:
                        similarity = 1.0
                    else:
                        similarity = max(0.0, 1.0 - (distance * penalty_per_edit))
                        # Track max non-initial similarity
                        max_non_initial_similarity = max(max_non_initial_similarity, similarity)
                    is_initial_match = False
                
                print(f"  Comparing '{token}' with '{cand_token}': similarity = {similarity:.3f}{' (initial match)' if is_initial_match else ''}")
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_index = i
                    best_distance = distance if not is_initial_match else 0  # Initial matches don’t use distance
                    best_candidate = cand_token
                    best_is_initial_match = is_initial_match
            
            if best_index is not None:
                print(f"Best match for '{token}' is '{larger_tokens[best_index]}' with similarity {best_similarity:.3f}")
                matched_indices.append(best_index)
                if best_distance > 0:  # Non-exact match
                    all_exact_matches = False
                    if not best_is_initial_match and token and best_candidate and token[0].lower() != best_candidate[0].lower():
                        letter_penalty = start_letter_penalty
                        best_similarity = max(0.0, best_similarity - letter_penalty)
                        print(f"  Start letter mismatch ('{token[0]}' vs '{best_candidate[0]}'), applying penalty: {letter_penalty:.3f}, adjusted similarity: {best_similarity:.3f}")
                # Adjust for initial match: only use if no strong non-initial match
                if best_is_initial_match and max_non_initial_similarity >= threshold:
                    # Revert to best non-initial match if it exists and exceeds threshold
                    for i, cand_token in enumerate(larger_tokens):
                        if i in matched_indices and i != best_index:
                            continue
                        distance = self.edit_distance(token.lower(), cand_token.lower())
                        similarity = 1.0 if distance == 0 else max(0.0, 1.0 - (distance * penalty_per_edit))
                        if similarity > best_similarity and similarity >= threshold:
                            best_similarity = similarity
                            best_index = i
                            best_candidate = cand_token
                            best_is_initial_match = False
                            matched_indices[-1] = best_index  # Update the match
                            print(f"  Overriding initial match with stronger non-initial match '{cand_token}' at {best_similarity:.3f}")
                            break
                total_similarity += best_similarity
            else:
                print(f"No match found for token '{token}'")
                all_exact_matches = False
                total_similarity += best_similarity
        
        # Base similarity based on smaller token set
        base_similarity = total_similarity / len(smaller_tokens) if smaller_tokens else 0.0
        
        # Penalize unmatched tokens in the larger set, but skip if all smaller tokens match exactly
        num_unmatched = len(larger_tokens) - len(matched_indices)
        unmatched_penalty_total = 0.0
        if not all_exact_matches:
            unmatched_penalty_total = num_unmatched * unmatched_penalty
        print(f"Number of unmatched tokens: {num_unmatched}, Unmatched penalty: {unmatched_penalty_total:.3f}")
        
        # Final score
        final_similarity = max(0.0, base_similarity - unmatched_penalty_total)
        print(f"Base similarity: {base_similarity:.3f}")
        print(f"Final similarity score: {final_similarity:.3f}")
        return round(final_similarity, 3)

    
    
    def apply_phase1_filters(self, query: Dict, candidate: Dict) -> bool:
        for algorithm in self.phase1_algorithms:
            min_score = algorithm['version']['min_score']
            func = getattr(self, algorithm['name'].lower().replace(' ', '_'))
            if func(query, candidate) < min_score:
                print(f"Failed {algorithm['name']} filter")
                return False
        print("Passed all Phase 1 filters")
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
            if algorithm['version'].get('terminates_on_exact_match', False) and score >= threshold:
                print(f"Threshold met by {algorithm['name']}, terminating scoring with score {score:.3f}")
                return score, scores  # Use this score directly instead of weighted average
            
            total_score += score * weight
            total_weight += weight
        final_score = total_score / total_weight if total_weight > 0 else 0.0
        print(f"No algorithm met threshold, using weighted average: {final_score:.3f}")
        return final_score, scores

    def find_matches(self, query_name: Dict) -> List[Dict]:
        matches = []
        print(f"\nProcessing query name: {query_name['original_name']}")
        print(f"\nMax number of matches: {self.max_number_of_matches}")
        
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
                    print(f"Reached maximum number of matches ({self.max_number_of_matches}). Stopping search.")
                    return matches

        # Step 2: Phase 1 filtering
        phase1_candidates = []
        for candidate in self.names:
            if self.apply_phase1_filters(query_name, candidate):
                phase1_candidates.append(candidate)

        # Step 3 & 4: Phase 2 scoring
        for candidate in phase1_candidates:
            print(f"\nEvaluating candidate: {candidate['original_name']}")
            final_score, algorithm_scores = self.calculate_phase2_scores(query_name, candidate)
            print(f"Final score: {final_score:.3f}")

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
                print(f"Reached maximum number of matches ({self.max_number_of_matches}). Stopping search.")
                break

        # Return only one match since max_number_of_matches is set to 1 for mv003
        return matches[0] if matches else None
    
    def get_phase_1_algorithms(self) -> List[str]:
        return self.phase1_algorithms
    
    def get_phase_2_algorithms(self) -> List[str]:
        return self.phase2_algorithms