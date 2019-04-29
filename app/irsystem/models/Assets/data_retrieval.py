import json
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
import numpy as np

# coding=utf-8

def scrape_extra_data(dataset_name, country, country_abbrev):
    korean_data = pd.read_csv(os.path.join(dataset_name))
    drama_titles = korean_data['Title']
    korean_data_new = korean_data
    korean_data_new['Year'] = np.nan
    korean_data_new['Rating'] = np.nan
    korean_data_new['Rating'] = np.nan
    korean_data_new['Metascore'] = np.nan
    korean_data_new['Votes'] = np.nan
    korean_data_new['Runtime'] = np.nan
    korean_data_new['Actors']= np.nan
    drama_titles = korean_data['Title']
    korean_data_new = korean_data
    korean_summaries = {}
    korean_user_reviews = {}
    num_summaries = 0
    num_reviews = 0

    for index, title in drama_titles.items():
        title = title.lower()
        title = title.strip()
        title = re.sub(r'\([^)]*\)', '', title)
        title = title.replace('â€™', "'")
        title = title.replace('â€', "'")
        title = title.replace ("'“", "")
        if "/" in title:
            ind = title.find("/")
            title = title[ind+1:]
        url = "https://www.imdb.com/search/title?country_of_origin=" + country_abbrev + "&title=" + title
        print(index)
        print(url)
        response = get(url)
        html_soup = BeautifulSoup(response.text, 'html.parser')
        if html_soup is not None:
            drama = html_soup.find('div', class_ = 'lister-item mode-advanced')
            if drama is not None:
                year = drama.h3.find('span', class_ = 'lister-item-year text-muted unbold')
                if year is not None:
                    year = year.text
                    year = re.sub('[^0-9]', '', year)
                rating = drama.strong
                if rating is not None:
                    rating = rating.text
                mscore = drama.find('span', class_ = 'metascore favorable')
                if mscore is not None:
                    mscore = int(mscore.text)
                votes = drama.find('span', attrs = {'name':'nv'})
                if votes is not None:
                    votes = int((votes['data-value']))
                runtime = drama.find('span', class_ = 'runtime')
                if runtime is not None:
                    runtime = runtime.text.strip()
                korean_data_new['Year'].iloc[index] = year
                korean_data_new['Rating'].iloc[index] = rating
                korean_data_new['Metascore'].iloc[index] = mscore
                korean_data_new['Votes'].iloc[index] = votes
                korean_data_new['Runtime'].iloc[index] = runtime
        try:
            html_soup = BeautifulSoup(response.text, 'html.parser')
            actors = html_soup.find('div', class_ = 'lister-item-content')
            actors = actors.find_all('p')[2]
            actors = actors.find_all('a')
            actors = [actor.text for actor in actors]
            actors = ", ".join(actors)
            korean_data_new.loc[index, 'Actors'] = actors
        except:
            korean_data_new['Actors'].iloc[index] = np.nan
        try:
            html_soup = BeautifulSoup(response.text, 'html.parser')
            drama = html_soup.find('div', class_ = 'lister-item mode-advanced')
            url2_tag = str(drama.h3.a)
            url2_tag = url2_tag[16:]
            url2_tag = url2_tag.partition('/">')[0]
            url2 = 'https://www.imdb.com/title/' + url2_tag + '/plotsummary?ref_=tt_stry_pl'
            response2 = get(url2)
            html_soup2 = BeautifulSoup(response2.text, 'html.parser')
            summaries = html_soup2.find('div', id = 'main')
            summaries = summaries.find('ul', id = "plot-summaries-content")
            summaries = summaries.find_all('p')
            summaries = [summary.text for summary in summaries]
            korean_summaries[index] = summaries
            if len(summaries) > 0:
                num_summaries = num_summaries + 1
        except:
            korean_summaries[index] = []
        try:
            html_soup = BeautifulSoup(response.text, 'html.parser')
            drama = html_soup.find('div', class_ = 'lister-item mode-advanced')
            url3_tag = str(drama.h3.a)
            url3_tag = url3_tag[16:]
            url3_tag = url3_tag.partition('/">')[0]
            url3 = 'https://www.imdb.com/title/' + url3_tag + '/reviews?ref_=tt_urv'
            response3 = get(url3)
            html_soup3 = BeautifulSoup(response3.text, 'html.parser')
            reviews = html_soup3.find('div', class_ = 'lister-list')
            reviews = reviews.find_all('div', class_="text show-more__control")
            reviews = [review.text for review in reviews]
            korean_user_reviews[index] = reviews
            if len(reviews) > 0:
                num_reviews = num_reviews + 1
        except:
            korean_user_reviews[index] = []
    korean_data_new = korean_data_new.to_csv(os.path.join(country + '_data.csv'), encoding='utf-8')
    with open(country + '_summaries.json', 'w') as fp:
        json.dump(korean_summaries, fp)
    with open(country + '_user_reviews.json', 'w') as fp2:
        json.dump(korean_user_reviews, fp2)
    print("Number of Reviews Attained: " + str(num_reviews))
    print("Number of Summaries Attained: " + str(num_summaries))
scrape_extra_data("American_data.csv", "American", "us")
scrape_extra_data("korean_data.csv", "korean", "kr")
