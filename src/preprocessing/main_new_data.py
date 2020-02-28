from src.Fake_News.Backend.database import connection
from src.preprocessing.MyBasicNLP import MyBasicNLP as NLP
from src.preprocessing.BlogText import BlogText
from src.preprocessing.LIWC import LIWC
from src.preprocessing.MyTextPosTags import MyTextPosTags as POS
from src.preprocessing.TextStatistics import TextStatistics as TS
import time
import sys
import sqlite3
import pandas as pd
import nltk


# Various Functions
def redundancy():
    my_count = LIWC().category_count(lexicon="Function", stemmed_tokens=text_tokens_stemmed_dictionary,
                                     non_stemmed_tokens=text_tokens_dictionary)
    try:
        result = float(my_count) / float(nlp.get_filtered_words_count())
        return result
    except ZeroDivisionError:
        return 0


def content_word_diversity():
    try:
        return float(len(text_tokens_dictionary)) / float(sum(text_tokens_dictionary.values()))
    except ZeroDivisionError:
        return 0


categories = ["Feel", "Hear", "See", "They", "Verb", "Insight", "Cause", "Discrep", "Tentat", "Certain", "Differ",
              "Affiliation", "Power", "Reward", "Risk", "Work", "Leisure", "Relig", "Money", "Posemo", "Negemo",
              "Affect", "FocusPast", "FocusFuture", "I", "We", "You", "SheHe", "Quant", "Compare", "Negate", "Swear",
              "Netspeak", "Interrog", "Conj", "Cogproc", "Social", "Space", "Incl", "Excl", "Motion", "Time", "Article",
              "Prep", "Pronoun"]

pos_tags = ["NN", "NNP", "PRP", "PRP$", "WP", "DT", "WDT", "CD", "RB", "UH", "VB", "JJ", "VBD", "VBG", "VBN", "VBP",
            "VBZ", "MD"]

total_dictionary_words = LIWC.get_dictionary_words(categories=categories)  # creates a set of words from the dictionary
# categories we used

t0 = time.time()  # Define starting time

if __name__ == "__main__":
    cnx = sqlite3.connect('F:\\zagkotsis\\newsALL.db')
    types = ['fake', 'satire', 'bias', 'conspiracy', 'junksci', 'hate', 'clickbait', 'unreliable', 'political',
             'reliable']
    for type in types:
        db = pd.read_sql_query("SELECT * FROM data where type='" + type + "' LIMIT 10", cnx)

        for index, record in enumerate(db):

            sys.stdout.write('\r')
            sys.stdout.write("Statement %i out of %i" % (index+1, db.count()))
            sys.stdout.flush()
            time.sleep(0.2)

            text = record['content']

            ts = TS(text)
            nlp = NLP(text=text)
            text_tokens_dictionary = nlp.get_normalized_tokens_as_dict()
            text_tokens_stemmed_dictionary = nlp.get_stemmed_tokens_as_dict()
            tokens = nlp.get_normalized_tokens()

            item = {"_id": record['index'],
                    "words": ts.get_words(),
                    "sentences": ts.get_sentences(),
                    "noun phrases": ts.noun_phrases_count(),
                    "Clauses": ts.get_clauses(),
                    "avg_words_per_sentence": ts.get_average_words_per_sentence(),
                    "avg_word_length": ts.get_average_word_length(),
                    "avg_noun_phrase_length": ts.noun_phrase_avg_length(),
                    "Pausality": ts.get_pausality(),
                    "Modifiers": POS(text_tokens_dictionary).get_modifiers(),
                    "dictionary words": LIWC.get_words_captured(tokens=tokens, dictionary_words=total_dictionary_words),
                    "Emotiveness": POS(text_tokens_dictionary).get_emotiveness(),
                    "lexical_word_diversity": ts.get_lexical_word_diversity(),
                    "content_word_diversity": content_word_diversity(),
                    "redundancy": redundancy(),
                    "errors": nlp.get_errors(),
                    "syllable_count": ts.get_syllables(),
                    "of_big_words": ts.get_long_words(),
                    "syllables_per_word": ts.get_average_syllables_per_word(),
                    "of_short_sentences": nlp.get_short_sentences(),
                    "of_long_sentences": nlp.get_long_sentences(),
                    "flesh_kincaid": ts.get_flesh_kincaid(),
                    # "Raa": POS(text_tokens_dictionary).get_rate_of_adjectives_adverbs(),
                    "fog_index": ts.get_fog_index(),
                    "smog_index": ts.get_smog_index(),
                    "avg_verb_phrase_length": ts.verb_phrase_avg_length(),
                    "flu_reuters_c": nlp.get_flu_coca_c(),
                    "flu_reuters_d": nlp.get_flu_coca_d(),
                    "stopwords_percent": nlp.stop_words_percent(),
                    "of_words_capital": ts.get_capital_words(),
                    "punctuation": ts.get_punctuation(),
                    "quotes": ts.get_quotes(),
                    "verb_phrases": ts.verb_phrases_count(),
                    "label": record['type']['type']}

            # Loop for LIWC features
            for category in categories:
                LIWC_count = LIWC().category_count(lexicon=category, stemmed_tokens=text_tokens_stemmed_dictionary,
                                                   non_stemmed_tokens=text_tokens_dictionary)
                item[category] = LIWC_count

            # Loop for pos tags features
            for tag in pos_tags:
                tag_counter = POS(text_tokens_dictionary).get_pos_tag(tag=tag)
                item[tag] = tag_counter





        print("Time needed: ", time.time() - t0)
        break