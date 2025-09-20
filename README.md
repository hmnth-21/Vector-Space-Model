**Vector Space Model **

This project implements a Vector Space Model (VSM) for Information Retrieval. Given a document corpus, the model represents documents and queries as vectors in a multi-dimensional space and measures similarity (e.g., using cosine similarity) to retrieve the most relevant documents.

Project Structure : 
- corpus – contains the collection of documents used for testing. - elite_vector_space_model.py – the driver script to load the corpus, build the vector space model, and handle queries. - Other helper files/modules as needed.
  
How to Run : 
1. Clone this repository    git clone https://github.com/hmnth-21/Vector-Space-Model.git cd Vector-Space-Model 
2. Install dependencies , If there is a requirements.txt: pip install -r requirements.txt Otherwise, install manually: pip install numpy nltk  
3. Run the program    python3 elite_vector_space_model.py 
4. Provide a query. When prompted, enter a search query. The program will compute similarity scores between the query and all documents in the corpus, then return the most relevant documents.

Features : 
- Preprocessing of corpus (tokenization, stopword removal, stemming/lemmatization if required). - Construction of a term-document matrix. - Query handling using the same vector space. - Similarity computation using cosine similarity. - Ranking of documents based on relevance.
  
Contributors : 
- Hemanth M S (Roll No: 2310110122)
- Pranav J (Roll No: 2210110460)
- Shanmuganathan Ramakrishnan (Roll No: 2310110273)

References : 
- Salton, G., Wong, A., & Yang, C. S. (1975). A vector space model for automatic indexing. Communications of the ACM. - NLP & Information Retrieval course materials.
