import requests
import os
from dotenv import load_dotenv
import json


def get_stories_from_url(date):
    """get a set of stories from Care Opinion API using a given URL
        Args:
            date: The date on which the stories posted.
        Returns:
    """
    try:
        load_dotenv()  # load the environmental variables to access the keys
        story_set = requests.get(
            "https://www.careopinion.org.uk/api/v2/opinions", params={"key": os.environ["CARE_OPINION_URL_KEY"],
                                                                      "publishedafter": date,
                                                                      "dateofpublicationorder": "ascending",
                                                                      "take": "200"})
        print(story_set.text)
    except:
        print("Something went wrong...")


def get_stories_from_http_request(date):
    """get a set of stories from Care Opinion API using a given API endpoint.
        Args:

        Returns:
    """
    try:
        load_dotenv()  # load the environmental variables to access the keys
        story_set = requests.get(
            "https://www.careopinion.org.uk/api/v2/opinions",
            headers={'Authorization': 'SUBSCRIPTION_KEY ' + os.environ["CARE_OPINION_API_KEY"]},
            params={"publishedafter": date,
                    "dateofpublicationorder": "ascending",
                    "take": "100"}
        )

        with open("../newDataset/"+date+".json", "w") as outfile:
            outfile.write(story_set.text)
    except Exception as e:
        print("Something went wrong...", e.__str__())


years = ["2020-"]
months = ["1-", "2-", "3-", "4-", "5-", "6-", "7-", "8-", "9-", "10-", "11-", "12-"]
days = ["1", "5", "10", "15", "20", "25"]

for year in years:
    for month in months:
        for day in days:
            get_stories_from_http_request(year+month+day)
