class Response:
    def __init__(self, oldId, old, new, visType, oldParse,userQuestion):
        self.oldId = oldId
        self.old = old
        self.new = new
        self.visType = visType
        self.oldParse = oldParse
        self.userQuestion = userQuestion

    def to_dict(self):
        return {
            "oldId": self.oldId,
            "old": self.old,
            "new": self.new,
            "visType": self.visType,
            "oldParse": self.oldParse,
        }