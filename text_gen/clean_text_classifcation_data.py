import pickle
import random

from nltk import word_tokenize, download
download("punkt_tab")
download("stopwords")

def clean_data(dataset):
    for data in dataset:
        text = data["text"]
        # split into words

        tokens = word_tokenize(text)
        # convert to lower case
        tokens = [w.lower() for w in tokens]
        # remove punctuation from each word
        import string

        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]
        # filter out stop words
        from nltk.corpus import stopwords

        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]

        data["words"] = words

    return dataset

with open("text_classification_train_raw", "rb") as f:
    train_data = pickle.load(f)
with open("text_classification_test_raw", "rb") as f:
    test_data = pickle.load(f)

# see some random data
random0 = random.randint(0, len(train_data) - 1)
random1 = random.randint(0, len(train_data) - 1)
random2 = random.randint(0, len(test_data) - 1)
print("Some random data:")
print(train_data[random0]["text"])
print("================================================")
print(train_data[random1]["text"])
print("====================TEST========================")
print(test_data[random2]["text"])

train_data = clean_data(train_data)
test_data = clean_data(test_data)

print("\n\n\n")
print("Some clean random data:")
print(train_data[random0]["words"])
print("================================================")
print(train_data[random1]["words"])
print("====================TEST========================")
print(test_data[random2]["words"])


with open("text_classification_train_words", "wb") as text_classification_train_file:
    pickle.dump(train_data, text_classification_train_file)

with open("text_classification_test_words", "wb") as text_classification_test_file:
    pickle.dump(test_data, text_classification_test_file)