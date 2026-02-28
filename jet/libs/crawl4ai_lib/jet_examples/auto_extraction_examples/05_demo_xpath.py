from crawl4ai import JsonXPathExtractionStrategy

xpath_schema = {
    "name": "News Articles with XPath",
    "baseSelector": "//article[@class='news-item']",
    "fields": [
        {"name": "headline", "selector": ".//h2[contains(@class, 'headline')]", "type": "text"},
        {"name": "author", "selector": ".//span[@class='author']/text()", "type": "text"},
        {"name": "publish_date", "selector": ".//time/@datetime", "type": "text"},
        {"name": "content", "selector": ".//div[@class='article-body']//text()", "type": "text"}
    ]
}

strategy = JsonXPathExtractionStrategy(xpath_schema, verbose=True)
