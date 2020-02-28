import scrapy
import os.path
import json
from pprint import pprint
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

process = CrawlerProcess(get_project_settings())

#This also works it fetches the urls

class QuotesSpiderLink(scrapy.Spider):
    name = "snopes-links"


    def start_requests(self):

        urls = ['https://www.snopes.com/tag/fake-news/']
        for x in range(1, 27):

            urls.append('https://www.snopes.com/tag/fake-news/page/'+str(x)+'/')

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for htmlPage in response.css('article'):

                yield {
                    'link': htmlPage.css('a::attr(href)').extract()

                }

        next_page = response.css('div.pagination-inner-wrapper  a::attr(href)').extract_first()

        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)
