import pickle
import os


def _read_data(folder, label):
    data = []

    files = os.listdir(folder)
    num_files = len(files)
    print(f"Preparing to read {num_files} files from {folder}")
    for i, filename in enumerate(files):
        if i % 1000 == 0:
            print(f"Read {i} files")
        review_id, rating = filename.replace(".txt", "").split('_')
        rating = int(rating)

        with open(folder + "/" + filename, "r", encoding="utf-8") as file:
            text = file.read()

        data.append({
            "id": review_id,
            "rating": rating,
            "label": label,
            "text": text
        })

    return data


def read_data(folder):
    return _read_data(folder + "/neg", 0) + _read_data(folder + "/pos", 1)

train_data = read_data("./stage_4_data/text_classification/train")
test_data = read_data("./stage_4_data/text_classification/test")

with open("text_classification_train_raw", "wb") as text_classification_train_file:
    pickle.dump(train_data, text_classification_train_file)

with open("text_classification_test_raw", "wb") as text_classification_test_file:
    pickle.dump(test_data, text_classification_test_file)

