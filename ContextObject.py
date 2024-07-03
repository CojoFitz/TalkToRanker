class ContextObject:
    def __init__(self):
        self.newId = []
        self.responses = []
        self.parsedInfo = []
        self.visType = []
        self.trackedVis = False
        self.trackedName = ''
        self.trackedResponses = {}
        self.subsetId = []
        self.isSubset = False
        self.fairFilter = False

    def to_dict(self):
        return {
            "newId": self.newId,
            "responses": [response.to_dict() for response in self.responses],
            "parsedInfo": self.parsedInfo
        }