from openai import OpenAI
import re
class matchResponse:
    def __init__(self, allFeats, apiKey):
        self.client = OpenAI(api_key=apiKey)
        self.allFeats = allFeats
    def reMatchFeature(self, feature):
        system_message = f"""You are an expert at spellcorrecting, reformatting, and matching. Your task is to take any feature given to you and attempt to match it to one of these features: {self.allFeats}. Output your response as only the name of the feature you match it with in plaintext as listed.
                Match to case sensitivity as well
                If you are not confident about a feature matching to anything, please just say NONE
           
             """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": 'You have been provided with a feature by the name of: ' + feature + 'Provide a better match based on the list provided'},
            ]
        )
        return response.choices[0].message.content
    def match_text(self, user_input, recentFeature, recentQuestion, recentResponse):
        system_message = f"""
        
        You are perfect at formatting messages, please format my messages to match the following task types as best as possible:

        1. "What is the distribution of {{features}}" - shows the distribution of the data. This one should also work for requests for mean, median, and standard deviation.
        2. "Show me the stability" - shows stability of the data
        3. "What are the most important features" - ranks feature attribution
        4. "What is the correlation of the target with {{features}}" - shows correlation between a feature and the target
        5. "(Filter or Subset) by {{number}}<{{feature}}<{{number}}" - filter OR subset based on a numerical feature range. You must specify for both the upper and lower bound, no deviations from the format allowed. You must type the one you want to do. Subsetting removes points not in the range, while filtering will just highlight the range.
        6. "(Filter or Subset) the data when {{feature}} is {{value}}" - filter OR subset based on the value of a categorical feature. You must type the one you want to do. Subsetting removes points not in the range, while filtering will just highlight the range.
        7. "Track the previous response as {{name}}" - tracks the previous response and stores its filters
        8. "Show me tracked response {{name}}" - shows the tracked response, allowing the user to compare how the response has changed in comparison to the current state.
        9. "What are the available features" - Lists the features that are available
        10. "Select the top {{number}}" - Filters the selection to be the top n items. An example would be "Select the top 8"
        11. "(Filter or Subset) the data such that {{feature}} is the most important feature" This creates a filter or subset to make sure that the columns selected are the most important feature.
        12. "Show me the fairness of the current filter on {{feature}}" Shows the fairness of the given feature
        13. "Suggest a better cutoff for fairness on {{feature}}" Suggests a better selection fo fairness of the given feature

        For instance, if I say "I would like to filter apples within the range of 30 and 90," match it to "Filter by 30<apple<90."

        For numerical filtering, use the format: "(Filter or Subset) by {{number}}<{{feature}}<{{number}}".
        For categorical filtering, use the format: "(Filter or Subset) the data when {{feature}} is {{value}}".

        If there is an area where the feature is not clearly specified, please utilize the most recent feature of: {recentFeature}

        For example if someone said 
        "What is its correlation with the target?", which is a query that relies upon prior information. Please attempt to use the previous feature of {recentFeature}
        The previous feature will be {recentFeature}, so keep that in mind for queries that seem to make prior references. 
        
        If the feature that has been typed out is vague, attempt to match it with the features being used in the dataset, which include:
        {self.allFeats}.

        The user may also ask follow up questions based on the last message they sent which was: {recentQuestion}

        They received the response: {recentResponse}
        

        If a user lists off multiple features such as like this:
        "What is the distribution of apples and oranges"
        Format it as:
        "What is the distribution of apples,oranges"
        Distribution and Correlation support extra features.
        Extra features mentioned must be in a comma seperated order such as: "feat1,feat2,feat3..." if there is only one feature just format it as: "feat1"




        Do's:
        - Do follow the exact format for each task type.
        - Do ensure numerical features use numerical filtering format.
        - Do ensure categorical features use categorical filtering format.
        - Do use the available features list to match features correctly.
        - If the user says filter, always filter. If the user says subset, always subset. 
        - If there is not a clear specification of the feature, use the previous feature.
        - Correctly format extra features

        Don'ts:
        - Don't make up responses that don't fit the explicit format.
        - Don't mix numerical and categorical filtering formats.
        - Don't ignore the list of available features.
        - Don't include quotation marks in the outputs
        - Don't subset when the request is to filter.
        - Don't filter when the request is to subset.

        Good Examples:
        - User says: "I do not want apples that are older than 5 years"
        Response: "Filter by 0<age<5"

        - User says: "I would like to subset apples within the range of 30 and 90"
        Response: "Subset by 30<apple<90"

        - User says: "Show me the stability"
        Response: "Show me the stability"

        - User says: "I want to higlight the points when age is the most important feature"
        Response: "Filter the data such that age is the most important feature"

        
        - User says: "I want to subset the data when age is the most important feature"
        Response: "Subset the data such that age is the most important feature"

        - User says: "I want to track this message"
        Response: "Track the previous response as tracked"
        
        -User says: "I want to know the distribution of feature"
        Response: What is the distribution of feature

        -User says: "What is the correlation of feature with the target"
        Response: "What is the correlation of the target with feature" 


        Bad Examples:
        - User says: "I do not want apples that are older than 5 years"
        Response: "Filter the data when apple is younger than 5" (Incorrect because it should use numerical filtering for age)

        - User says: "I would like to subset apples within the range of 30 and 90"
        Response: "Filter by 30<apple<90" (Incorrect because the user asked for subsetting)

        - User says: "What is the distribution of CGPA"
        Response: "Show me the distribution" (Incorrect because you need to say "what is the distribution" do not deviate)
        
        -User says: "I want to filter out the information of people older than 26"
        Response: "Filter by age<26" (Incorrect because you must have an upper and lower bound. Filter by 0<age<26 is the correct way to handle this)
        Only reply with the matched format. Do not make up any responses that don't fit this explicit format.

        - User says: "Filter item between 98 and 115"
        Response: "Subset by 98<item<115" Response is bad because this request was to filter. Filter when asked. 

        -User says: "What is the correlation of feature with the target"
        Response: "What is the correlation of income with the target" This response is bad since it does not have the correct order. Always format this as "What is the correlation of the target with feature"
    
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input},
            ]
        )
        return response.choices[0].message.content
    def queryParser(self,query,parsedInfo, lastResp, lastQues):
        p1 = r'(Filter|Subset) by (-?\d+\.?\d*)<(\w+)<(-?\d+\.?\d*)' #Numerical
        p2 = r"What is the correlation of the target with (\w+(?:\s*,\s*\w+)*)"
        p3 = r"What is the distribution of (\w+(?:\s*,\s*\w+)*)"
        p4 =  r"Show me the stability"
        p5 =  r"What are the most important features"
        p6 = r"Track the previous response as (\w+)"
        p7 = r"Show me tracked response (\w+)"
        p8 = r"(Filter|Subset) the data when feature (\w+) is (\w+)" #categorical
        p9 = r"What are the available features" #Task 6
        p10 = r"Select the top (\d+)"
        p11 = r"(Filter|Subset) the data such that (\w+) is the most important feature" #important feature
        p12 = r"Show me the fairness of the current filter on (\w+)"
        p13 = r"Suggest a better cutoff for fairness for feature (\w+)"
        patterns = [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13]
        matchData = [-9, None, None, None, None, query, False] #[Task, Capture1, Capture2, Capture3, Capture4, query, gptParse]
        for p in range(0, len(patterns)):
            match = re.match(patterns[p], query)
            if match:
                matchData[0] = p+1
                for i in range(1,len(match.groups())+1):
                    matchData[i] = match.group(i)
                if(p == 1 or p == 2):
                    matchData[1] = matchData[1].replace(" ", "")
                    matchData[1] = matchData[1].split(',')
                return matchData
        previousFeature = 'NO PREVIOUS FEATURE AVAILABLE'
        if (len(parsedInfo)>0):
            previousFeature = parsedInfo[0]
        gptParse = self.match_text(query, previousFeature, lastQues,lastResp)
        gptParse = gptParse.replace('"', '') #Sometimes it'll add quotation marks to the parse. I am just going to remove them to make things easier
        matchData[5] = gptParse
        matchData[6] = True
        for p in range(0, len(patterns)):
            match = re.match(patterns[p], gptParse)
            if match:
                matchData[0] = p+1
                for i in range(1,len(match.groups())+1):
                    matchData[i] = match.group(i)
                if(p == 1 or p == 2):
                    matchData[1] = matchData[1].split(',')
                return matchData
        return[-9, None, None, None, None, query, False] #If no match can be made, we will return an error