# Quora Duplicate Detection


This project is based on the stanford paper that tried to detect identical questions from Quora using neural networks and text embedding. I focused on the siamese neural network model, as it proved to be successful in several papers that tried to run on this specific task.

The network recieves the embeddings of two sentences through glove vectors (spacy, 'english_lg'). It uses an lstm layer to create representations of both embeddings and then computes various similarity measures (Hadamard product, euclidean distance, absolute difference). It then feeds these inputs (along with sentence lengths) to a dense layer and finally to an output softmax layer.
