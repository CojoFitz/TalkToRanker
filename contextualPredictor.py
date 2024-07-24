from openai import OpenAI

class contextualPredictor:
    def __init__(self, allFeats, apiKey):
        self.client = OpenAI(api_key=apiKey)
        self.allFeats = allFeats

    def predictContextualUse(self):
        feats = self.allFeats + ', y'
        predictUse = f"""You are in charge of briefly (in one sentence at most) describing what a dataset is used for based on the features inside of it. Use only the features given to you. """
        response = self.client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
        {"role": "system", "content": predictUse},
        {"role": "user", "content": 'The only features in this dataset that you need to know are: ' + feats + '. Give me a brief explanation of what the dataset is used for'}
        ]
        )
        return response.choices[0].message.content
    
    def contextualizeScore(self, useContext):
        feats = self.allFeats + ', y'
        scoreContext = 'You are an expert at renaming y values in datasets. These y values are generated based on the datasets, usually as predictions. When given the list of features and some context, you always come back with a very short but descriptive name that encapsulates the use of y in a way understandable by anyone.'
        response = self.client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
        {"role": "system", "content": scoreContext},
        {"role": "user", "content": 'Here is the context of how this dataset is likely used: ' + useContext  + 'Here are the features used: ' + feats + ' Give a better name of y. '}
        ]
    )
        return response.choices[0].message.content

    def contextualizeElement(self):
        nameContext = 'You are an expert at figuring out what rows in datasets could be referring to. You can tell what they are. For example you can tell if rows refer to things like apples, dogs, oranges, people, workers, and anything else. You are very good at using the names of the features to make this choice. With this ability, you will only say an improved pluaralized name of what each row/element can be referred to. '
        response = self.client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
        {"role": "system", "content":nameContext },
        {"role": "user", "content": 'Here are the features used: ' + self.allFeats + ' Give an element name. '}
        ]
    )
        return response.choices[0].message.content