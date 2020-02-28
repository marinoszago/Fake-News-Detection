from unittest import TestCase
from src.preprocessing.TextStatistics import TextStatistics as TS


class TestTextStatistics(TestCase):

    def setUp(self):
        self.test_punc = TS('!#?&!!')

    # def tearDown(self):
    #     self.tearDownMyStuff()
    #
    # def test_noun_phrases(self):
    #     self.fail()
    #
    # def test_noun_phrases_count(self):
    #     self.fail()
    #
    # def test_get_noun_phrases_words(self):
    #     self.fail()
    #
    # def test_noun_phrase_avg_length(self):
    #     self.fail()
    #
    # def test_verb_phrases(self):
    #     self.fail()
    #
    # def test_verb_phrases_count(self):
    #     self.fail()
    #
    # def test_get_verb_phrases_words(self):
    #     self.fail()
    #
    # def test_verb_phrase_avg_length(self):
    #     self.fail()
    #
    # def test_get_clauses(self):
    #     self.fail()
    #

    # def test_get_average_syllables_per_word(self):
    #     self.fail()
    #
    # def test_get_average_words_per_sentence(self):
    #     self.fail()

    def test_get_exclamation_marks(self):
        exc_marks = self.test_punc.get_exclamation_marks()
        self.assertEqual(exc_marks, 3)

    def test_get_punctuation(self):
        punctuation = self.test_punc.get_punctuation()
        self.assertEqual(punctuation, 6)

    # def test_get_quotes(self):
    #     self.fail()
    #
    # def test_get_capital_words(self):
    #     self.fail()
    #
    # def test_get_average_word_length(self):
    #     self.fail()
    #
    # def test_get_pausality(self):
    #     self.fail()
    #
    # def test_get_lexical_word_diversity(self):
    #     self.fail()
