from unittest import TestCase
from src.preprocessing.MyBasicNLP import MyBasicNLP as NLP


class TestMyBasicNLP(TestCase):

    def setUp(self):
        self.a = NLP("This is sample.")

    def test_get_token_list(self):
        token_list = self.a.get_token_list()
        self.assertEqual(len(token_list), 4, "The tokenization is correct")

    def test_get_normalized_tokens(self):
        normalized_tokens = self.a.get_normalized_tokens()
        self.assertEqual(normalized_tokens[0], "this", msg="The normalization is correct")
        self.assertEqual(len(normalized_tokens), 3, msg="punctuation removed correctly")

    def test_remove_stopwords(self):
        tokens_without_stopwords = self.a.remove_stopwords()
        self.assertEqual(len(tokens_without_stopwords), 1)

    def test_get_stem_tokens_normalized(self):
        tokens_stemmed = self.a.get_stem_tokens_normalized()
        self.assertEqual(tokens_stemmed[2], 'sampl')
        self.assertEqual(tokens_stemmed[0], 'this')

    def test_get_normalized_tokens_as_dict(self):
        tokens_dict = self.a.get_normalized_tokens_as_dict()
        d2 = {"this": 1, "is": 1, "sample": 1}
        self.assertDictEqual(d1=tokens_dict, d2=d2)
    #
    # def test_get_stemmed_tokens_as_dict(self):
    #     self.fail()
    #
    # def test_get_filtered_as_dict(self):
    #     self.fail()
    #
    # def test_get_filtered_words_count(self):
    #     self.fail()
    #
    # def test_stop_words_percent(self):
    #     self.fail()
    #
    # def test_get_sentences_tokenized(self):
    #     self.fail()
    #
    # def test_get_short_sentences(self):
    #     self.fail()
    #
    # def test_get_long_sentences(self):
    #     self.fail()
    #
    # def test_get_errors(self):
    #     self.fail()
    #
    # def test_get_sentence_depth(self):
    #     self.fail()
    #
    # def test_get_flu_coca_c(self):
    #     self.fail()
    #
    # def test_get_flu_coca_d(self):
    #     self.fail()
    #
    # def test_least_common_values(self):
    #     self.fail()
