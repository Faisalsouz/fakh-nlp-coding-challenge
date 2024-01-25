import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


class NLPClassifier:
    def __init__(self, model_name="en_core_web_md"):
        # Load spaCy model with word vectors
        self.nlp = spacy.load(model_name)
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.lr = LogisticRegression(random_state=42, max_iter=1000)
        self.accuracy = None
        self.precision = None

    def load_data(self, file_path):
        # Reading CSV file via pandas
        self.df = pd.read_csv(file_path)
        #return self.df

    def split_data(self, test_size=0.2, random_state=1):
        # Split data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df['Processed_Text'], self.df['class'], test_size=test_size, random_state=random_state
        )

    def get_document_vectors(self, text):
        doc = self.nlp(text)
        # Average the word vectors to get a document vector
        return doc.vector

    def prepare_data(self):
        # Create new columns in the dataframe and apply the function to get the document vector
        X_train_df = self.X_train.to_frame()
        X_train_df['doc_vector'] = X_train_df['Processed_Text'].apply(self.get_document_vectors)
        X_test_df = self.X_test.to_frame()
        X_test_df['doc_vector'] = X_test_df['Processed_Text'].apply(self.get_document_vectors)

        # Prepare data for the model
        X_train_vectors = pd.DataFrame(X_train_df['doc_vector'].tolist(), index=X_train_df.index)
        X_test_vectors = pd.DataFrame(X_test_df['doc_vector'].tolist(), index=X_test_df.index)

        return X_train_vectors, X_test_vectors

    def train_model(self, X_train_vectors, y_train):
        # Train the classifier
        self.lr.fit(X_train_vectors, y_train)

    def evaluate_model(self, X_test_vectors, y_test):
        # Predict the labels of the test set
        y_pred = self.lr.predict(X_test_vectors)

        # Get performance metrics
        self.accuracy = metrics.accuracy_score(y_test, y_pred)
        self.precision = metrics.precision_score(y_test, y_pred, average='weighted')

    def print_results(self):
        print("Accuracy:", round(self.accuracy, 2))
        print("Precision:", round(self.precision, 2))

    def perform_topic_modeling(self, num_topics=5):
        # Tokenize the documents
        tokenized_docs = [doc.split() for doc in self.df['Processed_Text']]

        # Create a spaCy Doc object for each document
        doc_objects = list(self.nlp.pipe(self.df['Processed_Text']))

        # Extract named entities as topics
        topics = [[ent.text for ent in doc.ents] for doc in doc_objects]

        self.topics = topics

        # Print the topics
        print("Topics:")
        for i, topic in enumerate(topics[:num_topics]):
            print(f"Topic {i + 1}: {topic}")
# Usage
if __name__ == '__main__':

    classifier = NLPClassifier()
    classifier.load_data('processed_data.csv')
    classifier.split_data()
    X_train_vectors, X_test_vectors = classifier.prepare_data()
    classifier.train_model(X_train_vectors, classifier.y_train)
    classifier.evaluate_model(X_test_vectors, classifier.y_test)
    classifier.print_results()
    classifier.perform_topic_modeling()
