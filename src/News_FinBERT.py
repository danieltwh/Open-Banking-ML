import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
MODEL_PATH = (r"C:\Users\wangt\Open-Banking-ML\results\models\FinBERT_v1.0")
trained_bert = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

def evaluation(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, preds)}

def tokenize(examples):
    return tokenizer(examples['text'], truncation=True)

def predict(df):
    test_titles = df.title
    test_labels = df.label
    test_ds_raw = Dataset.from_dict({'text': test_titles, 'labels': test_labels})
    test_ds = test_ds_raw.map(tokenize, batched=False)
    trained_model = Trainer(trained_bert,tokenizer=tokenizer,)
    output = trained_model.predict(test_dataset=test_ds)
    encoder = LabelEncoder()
    df['label'] = encoder.fit_transform(df['label'])
    encoder.inverse_transform([np.argmax(i) for i in output.predictions])
    preds = [np.argmax(i) for i in output.predictions]
    print(accuracy_score(preds, test_labels))

if __name__ == "__main__":
    #testing the accuracy, should print 0.929
    df = pd.read_csv(r"C:\Users\wangt\Open-Banking-ML\data\raw\news\newsapiorg_labelled.csv", index_col=0)
    # 0 = postive, 1 = negative, 2 = neutral
    df.loc[df["label"] == 0, 'label'] = 2
    df.loc[df['label'] == 1, 'label'] = 0
    df.loc[df['label'] == -1, 'label'] = 1
    predict(df)

