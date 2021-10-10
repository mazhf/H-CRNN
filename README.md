Paper: Hierarchical Convolutional Recurrent Neural Network for Chinese Text Classification

Environments:
tensorflow 1.14.0
keras 2.2.5
gensim 4.1.0

Run order:
merge_data.py —>cuda_data.py —> split_data_label.py —>split_data.py —>
split_preprocess.py —>merge_split_clean_data.py —>word2vec.py—>
construct_dataset_embedding_matrix.py—>textcrnn_hier.py

