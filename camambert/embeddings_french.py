import pandas as pd
import numpy as np
import re
import torch
import spacy
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
import keras
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import nltk

batch_size=16
nltk.download('stopwords')
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ]', ' ', text)
    return text

def get_pos_tags(text):
    doc = nlp(text)
    return [token.pos_ for token in doc]

def tokenize_and_encode(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def get_bert_embeddings(text):
    encoded_input = tokenizer(text, return_tensors='pt', padding="max_length", max_length=512, truncation=True)
    encoded_input = {key: value.to('cuda') for key, value in encoded_input.items()}
    with torch.no_grad():
        output = bmodel(**encoded_input, output_hidden_states=True)
    return output.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()

def sentence_features(text):
    words = text.split()
    return len(words), sum(len(word) for word in words) / len(words) if words else 0

def train_svm_model(X_train, y_train):
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf']}
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def create_cnn_model(input_shape, num_classes):
    text_input_layer = Input(shape=input_shape)
    text_layer = Conv1D(256, 3, activation='relu')(text_input_layer)
    text_layer = MaxPooling1D(3)(text_layer)
    text_layer = Conv1D(256, 3, activation='relu')(text_layer)
    text_layer = MaxPooling1D(3)(text_layer)
    text_layer = Conv1D(256, 3, activation='relu')(text_layer)
    text_layer = MaxPooling1D(3)(text_layer)
    text_layer = Conv1D(256, 3, activation='relu')(text_layer)
    text_layer = MaxPooling1D(3)(text_layer)
    text_layer = Conv1D(256, 3, activation='relu')(text_layer)
    text_layer = MaxPooling1D(3)(text_layer)
    text_layer = GlobalMaxPooling1D()(text_layer)
    text_layer = Dense(256, activation='relu')(text_layer)
    output_layer = Dense(num_classes, activation='softmax')(text_layer)
    model = Model(text_input_layer, output_layer)
    return model

def train_cnn_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=128):
    callback_list = [
        EarlyStopping(
            patience=20,
            monitor='acc',
        ),
        ModelCheckpoint(
            monitor='val_loss',
            save_best_only=True,
            filepath='model/movie_sentiment_m1.h5',
        ),
        ReduceLROnPlateau(
            patience=1,
            factor=0.1,
        )
    ]
    
    model.compile(optimizer=RMSprop(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callback_list,
                        validation_data=(X_test, y_test))

    return model, history

def preprocess_data(df, LE, encoder, scaler, tokenizer, bmodel):
    df['text'] = df['text'].apply(clean_text)
    df['pos_tags'] = df['text'].apply(get_pos_tags)
    all_pos_tags = set(tag for tags in df['pos_tags'] for tag in tags)
    one_hot_vectors = []
    for tags in df['pos_tags']:
        vector = [1 if pos_tag in tags else 0 for pos_tag in all_pos_tags]
        one_hot_vectors.append(vector)
    pos_tags_df = pd.DataFrame(one_hot_vectors, columns=list(all_pos_tags))
    df = pd.concat([df, pos_tags_df], axis=1)

    text_batches = [df['text'][i:i + batch_size] for i in range(0, len(df), batch_size)]
    embeddings = []
    for batch in text_batches:
        embeddings.append(get_bert_embeddings(list(batch)))
    embeddings = np.concatenate(embeddings)
    flattened_embeddings = embeddings.reshape((len(df), -1))
    for i in range(flattened_embeddings.shape[1]):
        df[f'embed_{i}'] = flattened_embeddings[:, i]

    df['num_words'], df['avg_word_length'] = zip(*df['text'].apply(sentence_features))
    df['num_words_str'] = df['num_words'].astype(str)
    df['avg_word_length_str'] = df['avg_word_length'].astype(str)

    X = df.drop(['text', 'pos_tags', 'id', 'num_words_str', 'avg_word_length_str'], axis=1)
    X_val = scaler.transform(X)
    X_val_r = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    
    y_val = bmodel.predict(X_val_r)
    df['difficulty'] = LE.inverse_transform(encoder.inverse_transform(y_val))
    df.drop(['sentence'], axis=1, inplace=True)
    return df

if __name__ == "__main__":
    # Load the dataset
    file_path = '/content/training_data.csv'  # Replace with your file path
    data = pd.read_csv(file_path)
    data = data.rename(columns={'sentence': 'text', 'difficulty': 'labels'}).drop(['id'], axis=1)

    LE = LabelEncoder()
    data['labels'] = LE.fit_transform(data['labels'])
    
    french_stopwords = set(stopwords.words('french'))

    nlp = spacy.load('fr_core_news_sm')
    
    train_data, val_data = train_test_split(data, test_size=0.2)
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)
    
    tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    
    bmodel = AutoModelForSequenceClassification.from_pretrained("/content/drive/MyDrive/cambert_french_finetuned", num_labels=6)
    
    training_args = TrainingArguments(output_dir="test_trainer", num_train_epochs=19, evaluation_strategy="epoch")
    
    trainer = Trainer(
        model=bmodel,
        args=training_args,
        train_dataset=traintokenized_datasets,
        eval_dataset=valtokenized_datasets,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    trainer.save_model("/content/drive/roberta_french_finetuned/")
    
    frenc_texts = data['text'].values
    
    all_pos_tags = {tag for tag in nlp.get_pipe("morphologizer").labels}
    tag_dict = {tag: i for i, tag in enumerate(all_pos_tags)}
    
    df = preprocess_data(df, LE, encoder, scaler, tokenizer, bmodel)
    
    submit_df = pd.read_csv('https://raw.githubusercontent.com/DalipiDenis/assign/main/unlabelled_test_data.csv')
    submit_df['difficulty'] = LE.inverse_transform(encoder.inverse_transform(y_val))
    submit_df.drop(['sentence'], axis=1, inplace=True)
    submit_df.to_csv('submission_2.csv', index=False)
