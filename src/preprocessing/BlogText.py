from textacy import preprocess
import pandas as pd
import numpy as np
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class BlogText:
    """
    Parse and pre-process tweets

    """

    def __init__(self, text, stopword_removal=True):
        """
        @ initialization spacy's fix_bad_unicode method
        is employed for avoiding exceptions in further
        pre-processing
        :param text: A string with the text for pre-processing
        """

        self.text = preprocess.fix_bad_unicode(text=text, normalization='NFC')
        self.text = self.remove_stopwords(stopword_removal)

    def get_demojize_char(self):
        _text = list(self.text)
        temp_text = []
        for character in _text:
            chars = emoji.demojize(character)
            if len(chars) == 1:
                temp_text.append(character)
            else:
                for char in chars:
                    temp_text.append(char)

        return temp_text

    def get_one_hot_encode_text(self, columns=280):
        """
        This method encodes a tweet based in US ASCII character set. We use only the
        printable graphic characters. The result is a DataFrame of 95 rows (ASCII
        character set and the desired number of columns. e.g. 280 for a tweet.

        :return:  A dataframe with the encoded tweet
        """

        if not isinstance(columns, int):
            return print("Get_one_hot_encode_tweet Type error. Please give an integer")

        charset = [" ", "!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+",
                   ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7",
                   "8", "9", ":", ";", "<", "=", ">", "?", "@", "A", "B", "C",
                   "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
                   "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[",
                   "\\", "]", "^", "_", "`", "a", "b", "c", "d", "e", "f", "g",
                   "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
                   "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~"]

        temp = np.zeros(len(charset))
        text = self.get_demojize_char()

        text_length = self.get_text_length_with_emojis()
        difference = columns - text_length - 1
        my_flag = False
        if difference > 0:
            print("I have to append zeros")
            for i in range(difference):
                text.append(None)
        else:
            print("I have to cut some columns")
            my_flag = True

        for index, text_char in enumerate(text):
            if my_flag:
                if index == columns - 1:
                    break
            temp_array = np.zeros(len(charset))
            if text_char in charset:
                for index, char in enumerate(charset):
                    if text_char == char:
                        temp_array[index] = 1
                        temp = np.column_stack((temp, temp_array))
                        pass
            else:
                temp = np.column_stack((temp, temp_array))

        df = pd.DataFrame(temp, index=charset)
        return df

    def get_init_text_length(self):
        return len(self.text)

    def get_text_length_with_emojis(self):
        return len(self.get_demojize_char())

    def remove_stopwords(self, _status):
        if _status is True:
            print("I will remove stopwords")
            stop_words = set(stopwords.words('english'))
            word_tokens = word_tokenize(self.text)

            filtered_text = [word for word in word_tokens if not word in stop_words]

            return filtered_text

        else:
            print("I won't remove stopwords")
            return self.text

    def print_text(self):
        return self.text


