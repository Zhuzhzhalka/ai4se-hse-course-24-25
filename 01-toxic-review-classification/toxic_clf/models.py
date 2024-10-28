import numpy as np
import pandas as pd
import torch
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from statistics import mean, stdev
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from tqdm import tqdm

warnings.filterwarnings("ignore") # .fit() spams, not relevant

TEST_SIZE = 0.2
N_SPLITS =  10

class ClassicMLWrapper:
    def __init__(self, data):
        self.X, self.X_eval, self.y, self.y_eval = train_test_split(data["prepared"], data["is_toxic"], test_size=TEST_SIZE)

    def print(self):
        print("LogisticRegression,CountVectorizer:")
        print(f"f1: {self.f1}")
        print(f"10-fold CV scores: {self.scores}")
        print(f"mean f1: {self.scores.mean()}")
        print(f"confusion matrix: {self.confusion_matrix}")

    def work(self):
        X, y = self.X, self.y
        count_vectorizer = CountVectorizer()
        count_vectors = count_vectorizer.fit_transform(X)

        X_train_count, X_test_count, y_train_count, y_test_count = train_test_split(
            count_vectors, y, test_size=TEST_SIZE)

        simple_log_reg_count = LogisticRegression()
        simple_log_reg_count.fit(X_train_count, y_train_count)
        y_pred_count = simple_log_reg_count.predict(X_test_count)
        f1_count = f1_score(y_test_count, y_pred_count)
        self.f1 = f1_count

        kf = KFold(n_splits=N_SPLITS, shuffle=True)

        reg_count = simple_log_reg_count
        self.scores = cross_val_score(reg_count, count_vectors, y, cv=kf, scoring="f1")

        predictions = cross_val_predict(reg_count, count_vectors, y, cv=kf)
        self.confusion_matrix = confusion_matrix(y, predictions)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
        pipeline = Pipeline([("vectorizer", CountVectorizer()), ("classifier", LogisticRegression())])
        param_grid = {
            "vectorizer__min_df":       [1, 3],
            "vectorizer__max_features": [1000, 3000],
            "vectorizer__stop_words":   [None, "english"],
            "classifier__solver":       ["newton-cg", "newton-cholesky"],
            "classifier__C":            [0.001, 0.01, 0.1, 1, 10],
            "classifier__max_iter":     [100, 1000],
            "classifier__penalty":      [None, "elasticnet"],
        }

        self.grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring="f1")
        self.grid_search.fit(X_train, y_train)
        print(f"best score: {self.grid_search.best_score_}")
        print(f"best parameters: {self.grid_search.best_params_}")

        y_pred = self.grid_search.predict(self.X_eval)
        report = classification_report(self.y_eval, y_pred, output_dict=True)

        print("accuracy:", accuracy_score(self.y_eval, y_pred))
        print("precision:", report["weighted avg"]["precision"])
        print("recall:", report["weighted avg"]["recall"])
        print("f1:", report["weighted avg"]["f1-score"])


class UtilDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        elem = {key: value[i] for key, value in self.encodings.items()}
        elem["labels"] = torch.tensor(self.labels[i])
        return elem

class RobertaWrapper:
    def __init__(self, data):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            print("cuda device not available, fix amdgpu, backing to cpu for now")
            print("this little maneuver's gonna cost us 51 years")
            self.device = torch.device("cpu")

        self.output_dir = "./training_output"

        self.X, self.X_eval, self.y, self.y_eval = train_test_split(
            data["prepared"], data["is_toxic"], test_size=TEST_SIZE)

    def print(self):
        print(self.evals)

    def work(self):
        X, X_eval, y, y_eval = self.X, self.X_eval, self.y, self.y_eval
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaForSequenceClassification.from_pretrained("roberta-base").to(self.device)

        train_encodings = tokenizer(list(X), return_tensors="pt", truncation=True, padding=True)
        train_labels = list(y)
        eval_encodings = tokenizer(list(X_eval), return_tensors="pt", truncation=True, padding=True)
        eval_labels = list(y_eval)

        train_dataset = UtilDataset(train_encodings, train_labels)
        eval_dataset = UtilDataset(eval_encodings, eval_labels)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")

            print(f"confusion matrix: {confusion_matrix(labels, predictions)}")

            return {
                "accuracy": accuracy_score(labels, predictions),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics
        )

        trainer.train()

        self.evals = trainer.evaluate()

def classifier(dataset, model):
    data = dataset.to_pandas().dropna()

    if model == "classic_ml":
        classicML = ClassicMLWrapper(data)
        classicML.work()
        classicML.print()
    elif model == "roberta-base":
        robertaWrapper = RobertaWrapper(data)
        robertaWrapper.work()
        robertaWrapper.print()
    else:
        print("Error: unknown model")
        return
