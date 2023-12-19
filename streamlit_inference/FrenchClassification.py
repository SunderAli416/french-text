import streamlit as st

import spacy
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import pandas as pd
import torch
import joblib
import re
import numpy as np
import tensorflow as tf
#python -m spacy download fr_core_news_sm
def get_bert_embeddings(text):
    # Tokenize and encode the text

    tokenizer = AutoTokenizer.from_pretrained("camembert-base")

    bmodel = AutoModelForSequenceClassification.from_pretrained("model/cambert_french_finetuned", num_labels=6)
    bmodel.cuda()
    encoded_input =tokenizer(text, return_tensors='pt',padding="max_length", max_length=512,truncation=True)

    # Move encoded input to the device
    encoded_input = {key: value.to('cuda') for key, value in encoded_input.items()}

    # Get model output and extract the last hidden states
    with torch.no_grad():
        output = bmodel(**encoded_input,output_hidden_states=True)
        # print(output.keys())
    # Mean pooling
    return output.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()  # Move to CPU for compatibility with scikit-learn

def sentence_features(text):
    words = text.split()
    return len(words), sum(len(word) for word in words) / len(words) if words else 0

def get_pos_tags(text):

    nlp = spacy.load('fr_core_news_sm')  # Load the French model

    doc = nlp(text)
    return [token.pos_ for token in doc]

def classify_text(text):
    texts = [text]

# Create a DataFrame
    df = pd.DataFrame(texts, columns=['text'])

    def initial_clean(text):
      text = text.lower()
      text = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ]', ' ', text)
      return text

    df['text']=df['text'].apply(initial_clean)
    all_pos_tags= set(['ADJ','ADP','ADV','AUX','CCONJ','DET', 'INTJ', 'NOUN', 'NUM','PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SPACE', 'SYM', 'VERB', 'X'])
    # Apply the function to create a new column with POS tags
    print(df)
    df['pos_tags'] = df['text'].apply(get_pos_tags)
    one_hot_vectors = []
    for tags in df['pos_tags']:
        vector = [1 if pos_tag in tags else 0 for pos_tag in all_pos_tags]
        one_hot_vectors.append(vector)
    pos_tags_df = pd.DataFrame(one_hot_vectors, columns=list(all_pos_tags))

# Concatenate the POS tags DataFrame with your original DataFrame
    df = pd.concat([df, pos_tags_df], axis=1)
    embeddings = []
    embeddings.append(get_bert_embeddings(list(df['text'])))

    # Concatenate the embeddings
    embeddings = np.concatenate(embeddings)
    flattened_embeddings = embeddings.reshape((len(df), -1))

    # Assign the embeddings to the DataFrame
    for i in range(flattened_embeddings.shape[1]):
        df[f'embed_{i}'] = flattened_embeddings[:, i]

    df['num_words'], df['avg_word_length'] = zip(*df['text'].apply(sentence_features))
    loaded_scaler = joblib.load('weights/minmax_scaler.pkl')
    df=df[loaded_scaler.get_feature_names_out()]
    X=loaded_scaler.transform(df)
    model=tf.keras.models.load_model('weights/my_model.keras')
    X_r = X.reshape((X.shape[0], X.shape[1], 1))
    y_val = model.predict(X_r,verbose=False)
    encoder = joblib.load('weights/one_hot_encoder.pkl')
    LE = joblib.load('weights/label_encoder.pkl')
    label=LE.inverse_transform(encoder.inverse_transform(y_val))
    return label[0]



def main():
    st.title("French Text Difficulty Classification")
    input_text = st.text_area("Enter Text:", "")
    if st.button("Classify"):
        if input_text:
            classification_result = classify_text(input_text)
            st.write(f"Result: {classification_result}")
        else:
            st.write("Please enter some text before classifying.")
if __name__ == "__main__":
    main()
