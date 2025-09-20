

import os
import re
import math
from collections import defaultdict, Counter

class PorterStemmer:

    
    def __init__(self):
        self.vowels = "aeiou"
    
    def is_vowel(self, word, i):
        
        if i < 0 or i >= len(word):
            return False
        return word[i] in self.vowels or (word[i] == 'y' and i > 0 and not self.is_vowel(word, i-1))
    
    def measure(self, word):
        
        n = 0
        i = 0
        while i < len(word):
            
            while i < len(word) and not self.is_vowel(word, i):
                i += 1
            if i >= len(word):
                break
            i += 1
            
            while i < len(word) and self.is_vowel(word, i):
                i += 1
            if i >= len(word):
                break
            n += 1
            i += 1
        return n
    
    def contains_vowel(self, word):
        
        return any(self.is_vowel(word, i) for i in range(len(word)))
    
    def ends_with_double_consonant(self, word):
        
        if len(word) < 2:
            return False
        return (word[-1] == word[-2] and 
                not self.is_vowel(word, len(word)-1) and
                not self.is_vowel(word, len(word)-2))
    
    def cvc(self, word):
       
        if len(word) < 3:
            return False
        return (not self.is_vowel(word, len(word)-3) and
                self.is_vowel(word, len(word)-2) and
                not self.is_vowel(word, len(word)-1) and
                word[-1] not in 'wxy')
    
    def step1a(self, word):
       
        if word.endswith('sses'):
            return word[:-2]
        elif word.endswith('ies'):
            return word[:-2]
        elif word.endswith('ss'):
            return word
        elif word.endswith('s') and len(word) > 1:
            return word[:-1]
        return word
    
    def step1b(self, word):
        
        if word.endswith('eed'):
            stem = word[:-3]
            if self.measure(stem) > 0:
                return stem + 'ee'
            return word
        
        flag = False
        if word.endswith('ed'):
            stem = word[:-2]
            if self.contains_vowel(stem):
                word = stem
                flag = True
        elif word.endswith('ing'):
            stem = word[:-3]
            if self.contains_vowel(stem):
                word = stem
                flag = True
        
        if flag:
            if word.endswith(('at', 'bl', 'iz')):
                return word + 'e'
            elif self.ends_with_double_consonant(word) and word[-1] not in 'lsz':
                return word[:-1]
            elif self.measure(word) == 1 and self.cvc(word):
                return word + 'e'
        
        return word
    
    def step2(self, word):
        
        suffixes = {
            'ational': 'ate', 'tional': 'tion', 'enci': 'ence', 'anci': 'ance',
            'izer': 'ize', 'abli': 'able', 'alli': 'al', 'entli': 'ent',
            'eli': 'e', 'ousli': 'ous', 'ization': 'ize', 'ation': 'ate',
            'ator': 'ate', 'alism': 'al', 'iveness': 'ive', 'fulness': 'ful',
            'ousness': 'ous', 'aliti': 'al', 'iviti': 'ive', 'biliti': 'ble'
        }
        
        for suffix, replacement in suffixes.items():
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self.measure(stem) > 0:
                    return stem + replacement
                break
        return word
    
    def step3(self, word):
        
        suffixes = {
            'icate': 'ic', 'ative': '', 'alize': 'al', 'iciti': 'ic',
            'ical': 'ic', 'ful': '', 'ness': ''
        }
        
        for suffix, replacement in suffixes.items():
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self.measure(stem) > 0:
                    return stem + replacement
                break
        return word
    
    def step4(self, word):
    
        suffixes = [
            'al', 'ance', 'ence', 'er', 'ic', 'able', 'ible', 'ant', 'ement',
            'ment', 'ent', 'ion', 'ou', 'ism', 'ate', 'iti', 'ous', 'ive', 'ize'
        ]
        
        for suffix in suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self.measure(stem) > 1:
                    if suffix == 'ion' and stem and stem[-1] in 'st':
                        return stem
                    elif suffix != 'ion':
                        return stem
                break
        return word
    
    def step5(self, word):
        
        if word.endswith('e'):
            stem = word[:-1]
            m = self.measure(stem)
            if m > 1 or (m == 1 and not self.cvc(stem)):
                return stem
        
        if (word.endswith('l') and len(word) > 1 and 
            self.measure(word[:-1]) > 1 and word[-2] == 'd'):
            return word[:-1]
        
        return word
    
    def stem(self, word):
        
        if len(word) <= 2:
            return word
        
        word = word.lower()
        word = self.step1a(word)
        word = self.step1b(word)
        word = self.step2(word)
        word = self.step3(word)
        word = self.step4(word)
        word = self.step5(word)
        
        return word

class EliteVectorSpaceModel:
    
    
    def __init__(self):
        
        self.documents = {}
        self.vocabulary = set()
        self.term_frequencies = defaultdict(lambda: defaultdict(int))
        self.document_frequencies = defaultdict(int)
        self.document_lengths = {}
        self.total_documents = 0
        
        
        self.soundex_cache = {}
        self.phrase_frequencies = defaultdict(lambda: defaultdict(int))
        self.stemmer = PorterStemmer()
        
        # Stop words (common English words to filter out)
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'but', 'have', 'this', 'can',
            'had', 'would', 'there', 'we', 'what', 'your', 'when', 'him', 'my',
            'me', 'she', 'they', 'them', 'been', 'than', 'or', 'you', 'all',
            'any', 'each', 'no', 'some', 'such', 'only', 'own', 'so', 'now',
            'very', 'just', 'where', 'too', 'if', 'about', 'who', 'get', 'which'
        }
        
        
        self.document_vectors = {}
        
    def levenshtein_distance(self, s1, s2):
        
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def correct_spelling(self, word, max_distance=2):
        
        if word in self.vocabulary:
            return word
        
        best_match = word
        min_distance = max_distance + 1
        
        for vocab_word in self.vocabulary:
            if abs(len(word) - len(vocab_word)) <= max_distance:
                distance = self.levenshtein_distance(word, vocab_word)
                if distance <= max_distance and distance < min_distance:
                    min_distance = distance
                    best_match = vocab_word
        
        return best_match if min_distance <= max_distance else word
    
    def soundex(self, word):
       
        if not word or word in self.soundex_cache:
            return self.soundex_cache.get(word, "0000")
        
        word = word.upper().strip()
        if not word or not word.isalpha():
            self.soundex_cache[word] = "0000"
            return "0000"
        
        soundex = word[0]
        mapping = {
            'B': '1', 'F': '1', 'P': '1', 'V': '1',
            'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
            'D': '3', 'T': '3', 'L': '4', 'M': '5', 'N': '5', 'R': '6'
        }
        
        prev_code = mapping.get(word[0], '0')
        for char in word[1:]:
            code = mapping.get(char, '0')
            if code != '0' and code != prev_code:
                soundex += code
                if len(soundex) >= 4:
                    break
            prev_code = code
        
        soundex = soundex[:4].ljust(4, '0')
        self.soundex_cache[word] = soundex
        return soundex
    
    def advanced_tokenize(self, text):
        
        if not text:
            return []
        
        # Extract alphabetic tokens, convert to lowercase
        tokens = re.findall(r'[a-zA-Z]+', text.lower())
        
        
        processed_tokens = []
        for token in tokens:
            if len(token) >= 2 and token not in self.stop_words:
                stemmed = self.stemmer.stem(token)
                if stemmed and len(stemmed) >= 2:
                    processed_tokens.append(stemmed)
        
        return processed_tokens
    
    def extract_bigrams(self, tokens):
        
        bigrams = []
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i+1]}"
            bigrams.append(bigram)
        return bigrams
    
    def build_index(self, document_files):
        
        print("Building elite vector space model index...")
        print("Features: Porter Stemming + Stop Word Removal + Advanced Processing")
        
        
        for filename in document_files:
            if filename.endswith('.txt'):
                try:
                    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        self.documents[filename] = content
                except Exception as e:
                    print(f"Warning: Could not read {filename}: {e}")
        
        self.total_documents = len(self.documents)
        print(f"Loaded {self.total_documents} documents")
        
      
        for doc_id, content in self.documents.items():
            tokens = self.advanced_tokenize(content)
            term_counts = Counter(tokens)
            
            
            for term, count in term_counts.items():
                self.vocabulary.add(term)
                self.term_frequencies[term][doc_id] = count
            
           
            bigrams = self.extract_bigrams(tokens)
            bigram_counts = Counter(bigrams)
            for bigram, count in bigram_counts.items():
                self.phrase_frequencies[bigram][doc_id] = count
        
        
        for term in self.vocabulary:
            self.document_frequencies[term] = len(self.term_frequencies[term])
        
        
        for doc_id in self.documents.keys():
            length_squared = 0.0
            doc_vector = {}
            
            for term in self.vocabulary:
                if doc_id in self.term_frequencies[term]:
                    tf = self.term_frequencies[term][doc_id]
                    log_tf = 1.0 + math.log10(tf)
                    length_squared += log_tf * log_tf
                    doc_vector[term] = log_tf
            
            self.document_lengths[doc_id] = math.sqrt(length_squared)
            
            if self.document_lengths[doc_id] > 0:
                for term in doc_vector:
                    doc_vector[term] /= self.document_lengths[doc_id]
            self.document_vectors[doc_id] = doc_vector
        
        print(f"Index built: {len(self.vocabulary)} unique terms after stemming")
        print(f"Document vectors computed for pseudo-relevance feedback")
    
    def pseudo_relevance_feedback(self, original_query_vector, top_doc_id, alpha=1.0, beta=0.5):
        
        if not top_doc_id or top_doc_id not in self.document_vectors:
            return original_query_vector
        
        
        top_doc_vector = self.document_vectors[top_doc_id]
        
       
        
        refined_query = {}
        
       
        for term, weight in original_query_vector.items():
            refined_query[term] = alpha * weight
        
        
        for term, weight in top_doc_vector.items():
            if term in refined_query:
                refined_query[term] += beta * weight
            else:
                refined_query[term] = beta * weight
        
        norm = math.sqrt(sum(w * w for w in refined_query.values()))
        if norm > 0:
            for term in refined_query:
                refined_query[term] /= norm
        
        return refined_query
    
    def search(self, query, use_feedback=True, use_spelling_correction=True):
        
        
        query_tokens = self.advanced_tokenize(query)
        if not query_tokens:
            return []
        
        
        if use_spelling_correction:
            corrected_tokens = []
            for token in query_tokens:
                corrected = self.correct_spelling(token)
                corrected_tokens.append(corrected)
            query_tokens = corrected_tokens
        
        
        enhanced_tokens = list(query_tokens)
        for token in query_tokens:
            token_soundex = self.soundex(token)
            for vocab_term in self.vocabulary:
                if (self.soundex(vocab_term) == token_soundex and 
                    vocab_term != token and len(vocab_term) > 2):
                    enhanced_tokens.append(vocab_term)
        
        
        if len(query_tokens) >= 2:
            bigrams = self.extract_bigrams(query_tokens)
            enhanced_tokens.extend(bigrams)
        
        
        query_tf = Counter(enhanced_tokens)
        query_vector = {}
        query_norm_squared = 0.0
        
        for term, tf in query_tf.items():
            if term in self.vocabulary or term in self.phrase_frequencies:
                log_tf = 1.0 + math.log10(tf)
                
                if term in self.vocabulary:
                    df = self.document_frequencies[term]
                else:
                    df = len(self.phrase_frequencies[term])
                
                idf = math.log10(self.total_documents / df) if df > 0 else 0.0
                weight = log_tf * idf
                query_vector[term] = weight
                query_norm_squared += weight * weight
        
       
        query_norm = math.sqrt(query_norm_squared)
        if query_norm > 0:
            for term in query_vector:
                query_vector[term] /= query_norm
        
        
        initial_scores = self._calculate_similarities(query_vector)
        
       
        if use_feedback and initial_scores:
            top_doc_id = initial_scores[0][0]
            refined_query = self.pseudo_relevance_feedback(query_vector, top_doc_id)
           
            final_scores = self._calculate_similarities(refined_query)
            return final_scores
        
        return initial_scores
    
    def _calculate_similarities(self, query_vector):
       
        doc_scores = []
        for doc_id in self.documents.keys():
            similarity = 0.0
            doc_length = self.document_lengths.get(doc_id, 0.0)
            
            if doc_length > 0:
                for term, query_weight in query_vector.items():
                    doc_tf = 0
                    
                    if term in self.vocabulary and doc_id in self.term_frequencies[term]:
                        doc_tf = self.term_frequencies[term][doc_id]
                    elif term in self.phrase_frequencies and doc_id in self.phrase_frequencies[term]:
                        doc_tf = self.phrase_frequencies[term][doc_id]
                    
                    if doc_tf > 0:
                        doc_weight = (1.0 + math.log10(doc_tf)) / doc_length
                        similarity += query_weight * doc_weight
            
            if similarity > 0:
                doc_scores.append((doc_id, similarity))
        
       
        doc_scores.sort(key=lambda x: (-x[1], x[0]))
        return doc_scores[:10]
    
    def format_results(self, results):
        """Format results for output"""
        formatted = []
        for doc_id, similarity in results:
            formatted.append(f"{doc_id}, {similarity}")
        return formatted

def main():
    
    print("=" * 70)
    print("ELITE VECTOR SPACE MODEL FOR INFORMATION RETRIEVAL")
    print("=" * 70)
    print("Advanced Features:")
    print("• Porter Stemming + Stop Word Removal")
    print("• Pseudo-Relevance Feedback (Rocchio Algorithm)")
    print("• Spelling Correction with Edit Distance")
    print("• Soundex matching + Phrase support")
    print()
    
    
    vsm = EliteVectorSpaceModel()
    
   
    text_files = [f for f in os.listdir('.') if f.endswith('.txt')]
    
    if not text_files:
        print("No text files found in current directory!")
        return
    
    vsm.build_index(text_files)
    
    print()
    print("=" * 70)
    print("ELITE SEARCH INTERFACE")
    print("=" * 70)
    print("Enter queries to search the document collection")
    print("Features: Auto spell-check, query refinement, stemming")
    print("Type 'quit' or 'exit' to terminate")
    print()
    
    
    while True:
        try:
            query = input("Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            
            results = vsm.search(query, use_feedback=True, use_spelling_correction=True)
            
            if not results:
                print("No relevant documents found.")
            else:
                print(f"\nTop {len(results)} results (with query refinement):")
                print("-" * 60)
                for i, (doc_id, similarity) in enumerate(results, 1):
                    print(f"{i:2d}. {doc_id}, {similarity:.16f}")
            
            print()
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def test_cases():
    
    print("RUNNING ELITE TEST CASES...")
    print("=" * 60)
    
   
    vsm = EliteVectorSpaceModel()
    text_files = [f for f in os.listdir('.') if f.endswith('.txt')]
    vsm.build_index(text_files)
    
    # Test Case 1
    print("\nTest Case 1: Zomato Business Query")
    query1 = "Developing your Zomato business account and profile is a great way to boost your restaurant's online reputation"
    results1 = vsm.search(query1)
    print(f"Query: {query1[:60]}...")
    print("Results:")
    for i, (doc_id, score) in enumerate(results1, 1):
        print(f"  {i}. {doc_id}, {score:.16f}")
    
    # Test Case 2
    print("\nTest Case 2: Shakespeare Biography Query")
    query2 = "Warwickshire, came from an ancient family and was the heiress to some land"
    results2 = vsm.search(query2)
    print(f"Query: {query2}")
    print("Results:")
    for i, (doc_id, score) in enumerate(results2, 1):
        print(f"  {i}. {doc_id}, {score:.16f}")
    
    # Test Case 3: 
    print("\nTest Case 3: Spelling Correction Demo")
    query3 = "Shakespere familie heiress"  # Misspelled words
    results3 = vsm.search(query3, use_spelling_correction=True)
    print(f"Query (with typos): {query3}")
    print("Results (auto-corrected):")
    for i, (doc_id, score) in enumerate(results3, 1):
        print(f"  {i}. {doc_id}, {score:.16f}")

if __name__ == "__main__":
    
    test_cases()
    print("\n" + "=" * 70)
    
   
    main()