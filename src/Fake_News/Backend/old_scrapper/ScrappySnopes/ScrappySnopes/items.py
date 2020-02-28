# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class ScrappysnopesItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass


class SnopesArticleInLanding(scrapy.Item):
    title = scrapy.Field()
    smallText = scrapy.Field()
    fakeNewsTag = scrapy.Field()
    visited = scrapy.Field()