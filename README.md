# Text Sentiment Analysis using Sequential Networks with Additive Attention & Transformers

<img src = "https://imerit.net/wp-content/uploads/2021/07/what-is-sentiment-analysis.jpg">

**Sentiment analysis** is an NLP technique to classify the sentiment (_positive_, _negative_, or _neutral_) of text data. It involves processing textual input, classifying sentiment using techniques like lexicon-based, machine learning, or deep learning models, and has applications in business, social media monitoring, finance, and healthcare. It helps understand public sentiment, customer satisfaction, and market trends.

## Binary Text Sentiment Analysis

## MultiClass Text Sentiment Analysis

> [!NOTE]
> `Small Spacy Embeddings` were used for this experiment. 

| Model            | Best Epoch | Train Loss | Test Loss | Train Acc | Test Acc |
|------------------|------------|------------|-----------|-----------|----------|
| LSTM + Attention | 23         | 0.5530     | 0.7040    | 0.7748    | 0.7124   |
| BiLSTM + Attention | 14       | 0.5572     | 0.6687    | 0.7680    | 0.7235   |
| GRU + Attention  | 21         | 0.5742     | 0.6618    | 0.7629    | 0.7250   |
| **BiGRU + Attention** | **22**        | **0.5099**     | **0.6593**    | **0.7893**    | **0.7335**   |

### Confusion Matrix

| LSTM + Attention | BiLSTM + Attention | GRU + Attention |BiGRU + Attention |
|------------------|------------|------------|-----------|
| ![conf_mat1](assets/lstm+attention.png) | ![conf_mat1](assets/bilstm+attention.png) | ![conf_mat1](assets/gru+attention.png)  |![conf_mat1](assets/bigru+attention.png) |





---
Feel free to send issues if you face any problem.</br>
Don't forget to star the repo :star: <img src="https://user-images.githubusercontent.com/74038190/213844263-a8897a51-32f4-4b3b-b5c2-e1528b89f6f3.png" width="25px" />