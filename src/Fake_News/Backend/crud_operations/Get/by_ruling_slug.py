class RulingSlug():

    def getPantsFireStatements(self, collection):
        pantsfire_statements = []
        for statement in collection.find({"ruling.ruling_slug": 'pants-fire'}):
            pantsfire_statements.append({
                "ruling_comments": statement["ruling_comments"],
                "ruling_comments_date": statement["ruling_comments_date"],
                "ruling_headline": statement["ruling_headline"],
                "ruling_date": statement["ruling_date"],
                "ruling_slug": statement["ruling"]["ruling_slug"]
            })

        return pantsfire_statements


    def getFullFlopStatements(self, collection):
        fullflop_statements = []
        for statement in collection.find({"ruling.ruling_slug": 'full-flop'}):
            fullflop_statements.append({
                "ruling_comments": statement["ruling_comments"],
                "ruling_comments_date": statement["ruling_comments_date"],
                "ruling_headline": statement["ruling_headline"],
                "ruling_date": statement["ruling_date"],
                "ruling_slug": statement["ruling"]["ruling_slug"]
            })

        return fullflop_statements

    def getNoFlipStatements(self, collection):
        noflip_statements = []
        for statement in collection.find({"ruling.ruling_slug": 'no-flip'}):
            noflip_statements.append({
                "ruling_comments": statement["ruling_comments"],
                "ruling_comments_date": statement["ruling_comments_date"],
                "ruling_headline": statement["ruling_headline"],
                "ruling_date": statement["ruling_date"],
                "ruling_slug": statement["ruling"]["ruling_slug"]
            })

        return noflip_statements

    def getHalfFlipStatements(self, collection):
        halfflip_statements = []
        for statement in collection.find({"ruling.ruling_slug": 'half-flip'}):
            halfflip_statements.append({
                "ruling_comments": statement["ruling_comments"],
                "ruling_comments_date": statement["ruling_comments_date"],
                "ruling_headline": statement["ruling_headline"],
                "ruling_date": statement["ruling_date"],
                "ruling_slug": statement["ruling"]["ruling_slug"]
            })

        return halfflip_statements

    def getFalseStatements(self, collection):
        false_statements = []
        for statement in collection.find({"ruling.ruling_slug": 'false'}):
            false_statements.append({
                "ruling_comments": statement["ruling_comments"],
                "ruling_comments_date": statement["ruling_comments_date"],
                "ruling_headline": statement["ruling_headline"],
                "ruling_date": statement["ruling_date"],
                "ruling_slug": statement["ruling"]["ruling_slug"]
            })

        return false_statements

    def getBarelyTrueStatements(self, collection):
        barelytrue_statements = []
        for statement in collection.find({"ruling.ruling_slug": 'barely-true'}):
            barelytrue_statements.append({
                "ruling_comments": statement["ruling_comments"],
                "ruling_comments_date": statement["ruling_comments_date"],
                "ruling_headline": statement["ruling_headline"],
                "ruling_date": statement["ruling_date"],
                "ruling_slug": statement["ruling"]["ruling_slug"]
            })

        return barelytrue_statements

    def getHalfTrueStatements(self, collection):
        halftrue_statements = []
        for statement in collection.find({"ruling.ruling_slug": 'half-true'}):
            halftrue_statements.append({
                "ruling_comments": statement["ruling_comments"],
                "ruling_comments_date": statement["ruling_comments_date"],
                "ruling_headline": statement["ruling_headline"],
                "ruling_date": statement["ruling_date"],
                "ruling_slug": statement["ruling"]["ruling_slug"]
            })

        return halftrue_statements

    def getMostlyTrueStatements(self, collection):
        mostlytrue_statements = []
        for statement in collection.find({"ruling.ruling_slug": 'mostly-true'}):
            mostlytrue_statements.append({
                "ruling_comments": statement["ruling_comments"],
                "ruling_comments_date": statement["ruling_comments_date"],
                "ruling_headline": statement["ruling_headline"],
                "ruling_date": statement["ruling_date"],
                "ruling_slug": statement["ruling"]["ruling_slug"]
            })

        return mostlytrue_statements

    def getTrueStatements(self, collection):
        true_statements = []
        for statement in collection.find({"ruling.ruling_slug": 'true'}):
            true_statements.append({
                "ruling_comments": statement["ruling_comments"],
                "ruling_comments_date": statement["ruling_comments_date"],
                "ruling_headline": statement["ruling_headline"],
                "ruling_date": statement["ruling_date"],
                "ruling_slug": statement["ruling"]["ruling_slug"]
            })

        return true_statements

    def getRulingSlagStatementCount(self, collection, annotation):

        counter = 0
        for statement in collection.find({"ruling.ruling_slug": annotation}, {"ruling.ruling_slug": annotation}):
            counter = counter + 1

        results = {"Annotation": annotation, "Counter": counter}

        return results
