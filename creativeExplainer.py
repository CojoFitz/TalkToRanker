from openai import OpenAI
import re

class creativeExplainer:
    def __init__(self,  predictedContext, apiKey):
        self.client = OpenAI(api_key=apiKey)
        self.predictedContext = predictedContext

    def explainGen(self,userQuestion, generatedResponse):

        explainInstructions = 'You are tasked with generating explanations based on questions relating to a dataset being used for a ranker model.  The dataset can be described as follows: ' + self.predictedContext + 'The explanations are for an interface where users can ask questions, this interface is called TalkToRanker, which is a visualization and conversational AI interface. You handle the conversational part for explainability to assist in decision making. You will be given a pre-generated response, alongside the user question. You must use the pre-generated response, the info you have on the dataset, alongside the users question to generate your response. Keep responses brief please.'
        response = self.client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
        {"role": "system", "content": explainInstructions},
        {"role": "user", "content": 'Question from user is: ' + userQuestion + ' Pre-generated response is: ' + generatedResponse }
        ]
    )
        return response.choices[0].message.content


