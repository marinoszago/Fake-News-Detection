class AdditionalInfo():
    def getAuthors(self, collection):
        authors = []
        for statement in collection.find():
            authors.append({"author": statement["author"]})

        return authors

    def getArts(self, collection):
        art = []
        for statement in collection.find():
            art.append({"art": statement["art"]})

        return art

    def getCanonicalURLs(self, collection):
        canonical_urls = []
        for statement in collection.find():
            canonical_urls.append({"canonical_url": statement["canonical_url"]})

        return canonical_urls

    def getCount(self, collection):
        count = 0
        for statement in collection.find():
            count += 1

        return count

    def getRulings(self, collection):
        rulings = []
        for statement in collection.find():
            rulings.append({
                "ruling_comments": statement["ruling_comments"],
                "ruling_comments_date": statement["ruling_comments_date"],
                "ruling_headline": statement["ruling_headline"],
                "ruling_date": statement["ruling_date"],
                "ruling_slug": statement["ruling"]["ruling_slug"]
            })

        return rulings

    def getStatementAndSlug(self, collection):
        statements = []
        for statement in collection.find():
            statements.append({
                "statement_id": statement['id'],
                "statement": statement['statement'],
                "statement_date": statement['statement_date'],
                "annotation": statement['annotation']
            })

        return statements

    def getStatementsByType(self, collection, statement_type):
        statements = []
        counter = 0
        for statement in collection.find({"statement_type.statement_type": statement_type},
                                         {'statement_type': 1, 'statement': 1, 'statement_date': 1}):
            statements.append(statement)
            counter = counter +1
        return counter