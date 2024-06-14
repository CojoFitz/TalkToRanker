from openai import OpenAI



from openai import OpenAI

class matchResponse:
    def __init__(self):
        self.client = OpenAI(api_key )
        self.system_message = f"""
        Please format my messages to match the following task types as best as possible:

        1. "Show me the data" - shows the distribution of the data
        2. "Show me the stability" - shows stability of the data
        3. "What are the most important features" - ranks feature attribution
        4. "What is the correlation of the target with {{feature}}" - shows correlation between a feature and the target
        5. "(Filter or Subset) by {{number}}<{{feature}}<{{number}}" - filter OR subset based on a numerical feature range. You must specify for both the upper and lower bound, no deviations from the format allowed. You must type the one you want to do. Subsetting removes points not in the range, while filtering will just highlight the range.
        6. "(Filter or Subset) the data when {{feature}} is {{value}}" - filter OR subset based on the value of a categorical feature. You must type the one you want to do. Subsetting removes points not in the range, while filtering will just highlight the range.
        7. "Track the previous response as {{name}}" - tracks the previous response and stores its filters
        8. "Show me tracked response {{name}}" - shows the tracked response, allowing the user to compare how the response has changed in comparison to the current state.


        For instance, if I say "I would like to filter apples within the range of 30 and 90," match it to "Filter by 30<apple<90."

        For numerical filtering, use the format: "(Filter or Subset) by {{number}}<{{feature}}<{{number}}".
        For categorical filtering, use the format: "(Filter or Subset) the data when {{feature}} is {{value}}".

        Do's:
        - Do follow the exact format for each task type.
        - Do ensure numerical features use numerical filtering format.
        - Do ensure categorical features use categorical filtering format.
        - Do use the available features list to match features correctly.

        Don'ts:
        - Don't make up responses that don't fit the explicit format.
        - Don't mix numerical and categorical filtering formats.
        - Don't ignore the list of available features.

        Good Examples:
        - User says: "I do not want apples that are older than 5 years"
        Response: "Filter by 0<age<5"

        - User says: "I would like to subset apples within the range of 30 and 90"
        Response: "Subset by 30<apple<90"

        - User says: "Show me the stability"
        Response: "Show me the stability"

        - User says: "I want to track this message"
        Response: "Track the previous response as tracked"

        Bad Examples:
        - User says: "I do not want apples that are older than 5 years"
        Response: "Filter the data when apple is younger than 5" (Incorrect because it should use numerical filtering for age)

        - User says: "I would like to subset apples within the range of 30 and 90"
        Response: "Filter by 30<apple<90" (Incorrect because the user asked for subsetting)

        - User says: "Show me the data distribution"
        Response: "Show me the stability" (Incorrect because it doesn't match the task type "Show me the data")
        
        -User says: "I want to filter out the information of people older than 26"
        Response: "Filter by age<26" (Incorrect because you must have an upper and lower bound. Filter by 0<age<26 is the correct way to handle this)
        Only reply with the matched format. Do not make up any responses that don't fit this explicit format.
        """
    def match_text(self, user_input):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content