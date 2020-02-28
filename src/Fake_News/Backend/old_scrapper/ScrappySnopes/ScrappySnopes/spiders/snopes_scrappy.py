import scrapy
import os.path
import json
from pprint import pprint


####This program is responsible for fetching the data such as title and small text and others from snopes
####Works OK!


try:
    mode = str(input('Do you want to save the html pages? (Y/N):'))
except ValueError:
    print
    "Not a valid response"



class LandingPageArticles(scrapy.Spider):
    name = "snopes"

    if (mode == "Y"):
        def save(self, response):
            page = response.url.split("/")[-2]
            filename = 'snopes-%s.html' % page
            with open(filename, 'wb') as f:
                f.write(response.body)

    def start_requests(self):
        urls = ['https://www.snopes.com/tag/fake-news/']
        for x in range(1, 27):

            urls.append('https://www.snopes.com/tag/fake-news/page/'+str(x)+'/')

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)



    def parse(self, response):
        for htmlPage in response.css('div.article-link-container'):

                yield {
                    'smallText': htmlPage.css('p.desc::text').extract(),
                    'title': htmlPage.css('h2.title::text').extract(),
                    'fakeNewsTag': htmlPage.css('div.breadcrumbs::text').extract(),
                    'article-date': htmlPage.css('span.article-date::text').extract(),
                    'seen': 'true'
                }

        next_page = response.css('div.pagination-inner-wrapper  a::attr(href)').extract_first()

        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)
