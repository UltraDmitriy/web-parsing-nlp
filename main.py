import csv
import itertools
import os
import time
from os import listdir

import nltk
import pdfplumber
from bs4 import BeautifulSoup
from natasha import NewsEmbedding, Segmenter, MorphVocab, NewsNERTagger, NamesExtractor, Doc, PER
from nltk import word_tokenize
from nltk.corpus import stopwords
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

URL = 'https://cyberleninka.ru/search?q=orm&page=1'
FILE = 'about-ORM.csv'


def get_current_url(current_page):
    current_url = f'https://cyberleninka.ru/search?q=orm&page={current_page}'
    return current_url


def get_html(url):
    option = webdriver.ChromeOptions()
    option.add_argument('headless')
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=option)
    driver.get(url)
    r = driver.page_source
    return r


def get_pages_count(html):
    soup = BeautifulSoup(html, 'html.parser')
    paginator = soup.find('ul', {"class": "paginator"})
    if paginator:
        return int(paginator.find_all('li')[-1].get_text())
    else:
        return 1


def get_content(html):
    articles = []

    soup = BeautifulSoup(html, 'html.parser')
    search_results_list = soup.find('ul', {'id': 'search-results'})
    for li in search_results_list.findAll('li'):
        name = li.find('h2', {'class': 'title'})
        link = name.find('a')
        author = li.find('span')
        span_block = li.find('span', {'class': 'span-block'})

        print('Заголовок статьи: ' + name.text)
        print('Ссылка: ' + 'https://cyberleninka.ru' + link['href'])
        print('Авторы статьи: ' + author.text)
        print('Год / Журнал: ' + span_block.text)

        articles.append({
            'title': name.text,
            'link': 'https://cyberleninka.ru' + link['href'],
            'authors': author.text,
            'year-journal': span_block.text,
        })
    return articles


def save_file(items, path):
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['Название', 'Ссылка', 'Авторы', 'Год/Журнал'])
        for item in items:
            writer.writerow([item['title'], item['link'], item['authors'], item['year-journal']])


def parse():
    html = get_html(URL)
    pages_count = get_pages_count(html)
    articles = []
    for page in range(1, pages_count + 1):
        print(f'Парсинг страницы {page} из {pages_count}...')
        html = get_html(get_current_url(page))
        articles.extend(get_content(html))
    save_file(articles, FILE)
    print(f'В CSV файл записано {len(articles)} статей')
    os.startfile(FILE)


def start_nlp_process():
    nltk.download('stopwords')
    keywords_joined = ''
    authors_joined = ''

    with open('about-ORM.csv', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)
        # for row in reader:
        for row in itertools.islice(reader, 3):
            option = webdriver.ChromeOptions()
            prefs = {"download.default_directory": os.path.dirname(os.path.abspath(__file__)) + '\pdfs'}
            option.add_argument('headless')
            option.add_experimental_option("prefs", prefs)
            driver = webdriver.Chrome(ChromeDriverManager().install(), options=option)
            driver.get(row[1])
            downloadpdf = driver.find_element_by_id('btn-download')
            downloadpdf.click()
            time.sleep(5)

            driver.close()

        for pdf_document in listdir(os.path.dirname(os.path.abspath(__file__))):
            if pdf_document.endswith(".pdf"):
                with open(pdf_document, "rb") as filehandle:
                    with pdfplumber.PDF(filehandle) as pdf:
                        pages = [page.extract_text() for page in pdf.pages]

                    text = '\n'.join(pages)
                    # print('=========================')
                    # print('Исходный текст')
                    # print(text)

                    punctuations = ['(', ')', ';', ':', '[', ']', ',', '—', '...', '.', '<', '>', '«', '»', '//', '-', '+', '-']
                    stop_words_ru = stopwords.words('russian')
                    stop_words_eng = stopwords.words('english')
                    tokens = word_tokenize(text)
                    all_except_stop_dict = nltk.FreqDist(
                        w.lower() for w in tokens if
                        w not in stop_words_ru and w not in stop_words_eng and w not in punctuations)

                    print(all_except_stop_dict.most_common(5))
                    # keywords_joined = ' '.join(str(key) for key, val in all_except_stop_dict.most_common(5))
                    print(keywords_joined)


                    emb = NewsEmbedding()
                    segmenter = Segmenter()
                    morph_vocab = MorphVocab()
                    names_extractor = NamesExtractor(morph_vocab)
                    ner_tagger = NewsNERTagger(emb)

                    doc = Doc(text)
                    doc.segment(segmenter)
                    doc.tag_ner(ner_tagger)

                    for span in doc.spans:
                        if span.type == PER:
                            print(span.text)



# parse()
start_nlp_process()
