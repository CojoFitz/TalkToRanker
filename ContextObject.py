class ContextObject:
    def __init__(self):
        self.newId = []
        self.responses = []
        self.parsedInfo = []
        self.visType = []

    def to_dict(self):
        return {
            "newId": self.newId,
            "responses": [response.to_dict() for response in self.responses],
            "parsedInfo": self.parsedInfo
        }