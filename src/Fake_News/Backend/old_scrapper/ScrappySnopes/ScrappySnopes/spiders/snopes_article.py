import scrapy
import os.path
import json
from pprint import pprint


class QuotesSpiderArticle(scrapy.Spider):
    name = "snopes-articles"


    def start_requests(self):
        links = []
        with open('../../ScrappySnopes/spiders/Resources/Data/articlesHrefs.json','r') as f:
            data = json.load(f)

            for link in data:
                links.append(str(link['link'][0]))

        for url in links:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for article in response.xpath('//div[@class="body-content"]/article'):
            if(response.xpath('//div[@class="body-content"]/article')!=None):
                yield {
                    'title': article.xpath('//header/h1[@class="article-title"]/text()').extract(),

                    'description': article.xpath('//header/h2[@class="article-description"]/text()').extract(),

                    'claim': article.xpath('//div[@class="article-text-inner"]'
                                           '/h3[@class="section-break claim" and ./span/text()=\'CLAIM\']'
                                           '/following-sibling::p[1]/text()').extract(),

                    'rating': article.xpath('//div[@class="article-text-inner"]'
                                            '/div[@class="rating-wrapper"]/a/span/text()').extract(),

                    'paragraphs': article.xpath('//div[@class="article-text-inner"]'
                                           '/h3[@class="section-break" or ./span/text()=\'ORIGIN\']'
                                           '/following-sibling::p/text()').extract(),

                    'paragraphsExtra': article.xpath('//div[@class="article-text-inner"]'
                                                '/p/following-sibling::p[not(a/img)]/text()').extract(),

                    'blockquotes': article.xpath('//div[@class="article-text-inner"]'
                                           '/h3[@class="section-break" and ./span/text()=\'ORIGIN\']'
                                           '/following-sibling::blockquote[p]/p/text()').extract(),

                    'innerLinksParagraph': article.xpath('//div[@class="article-text-inner"]'
                                           '/h3[@class="section-break" and ./span/text()=\'ORIGIN\']'
                                           '/following-sibling::p[a/@href and not(a/img)]/a/@href').extract(),

                    'innerLinksBlockquote': article.xpath('//div[@class="article-text-inner"]'
                                                 '/h3[@class="section-break" and ./span/text()=\'ORIGIN\']'
                                                 '/following-sibling::blockquote[a/@href]/a/@href').extract()

                }




