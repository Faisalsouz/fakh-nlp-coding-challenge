import pandas as pd
import json
import spacy
nlp = spacy.load("en_core_web_sm")

class PreProcessing:
    def __init__(self,file_path,label=None):
        self.lablel = label
        self.file_path = file_path
        loaded_data = self.load_data()
        relevant_data = self.extract_relevant_data(loaded_data)
        self.df = self.create_dataframe(relevant_data)

    def load_data(self):
        with open(self.file_path, 'r',encoding='utf-8') as f:
            data = json.load(f)
        return data

    def extract_relevant_data(self,data):
        articles = data['hits']['hits']
        articles_list = []
        for article in articles:
            article_id = article['_id']
            full_text = article['_source']['fullText']
            # append the article id and full text as a list to the articles_list
            articles_list.append({'Article_ID': article_id, 'Full_Text': full_text,'class': self.lablel})
        return articles_list

    def create_dataframe(self,articles_list):
        df = pd.DataFrame(articles_list)
        return df
    # join multiple dataframes
    def join_dataframes(self,df1,df2):
        df = pd.concat([df1,df2])
        return df

    def preprocess_text(self, text):
        # Tokenize the text using spaCy
        doc = nlp(text)

        # Lemmatize each token and convert to lowercase
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]

        # Join the tokens back into a single string
        processed_text = ' '.join(tokens)

        return processed_text



# if __name__ == '__main__':
#
#     df1 = PreProcessing('./data_query_from_9367.json','9367')
#     df2 = PreProcessing('./data_query_from_9578.json','9578')
#     joined_df = df1.join_dataframes(df1.df,df2.df)
#     joined_df['Processed_Text'] = joined_df['Full_Text'].apply(df1.preprocess_text)
#     joined_df.to_csv('processed_data.csv', index=False)
#     print(joined_df.head())
