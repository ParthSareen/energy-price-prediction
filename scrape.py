import requests
from bs4 import BeautifulSoup
import os
import re

# The base URL of the website to scrape
# base_url = "http://reports.ieso.ca/public/RealtimeConstTotals/"
# base_url = 'http://reports.ieso.ca/public/DispUnconsHOEP/'
base_url = 'http://reports.ieso.ca/public/WeeklyMarket/'

# Make a request to the website
response = requests.get(base_url)
response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code

# Parse the HTML content of the page with BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Define a regular expression pattern to match CSV files from 2023 onwards
pattern = re.compile(r'.*_(2023|202[4-9]|20[3-9]\d|2[1-9]\d{2})\d{4}\.csv')
pattern_2023 = re.compile(r'.*_(2023)\d{4}\.csv')
pattern_2024 = re.compile(r'.*_(2024)\d{4}\.csv')
# Find all the links on the page that match the pattern for 2023 or above
csv_links = [link.get('href') for link in soup.find_all('a', href=pattern)]

# Create a directory to store the downloaded files
os.makedirs('data/weekly_market2', exist_ok=True)

# Download each CSV file that matches the pattern
for link in csv_links:

    csv_url = base_url + link
    csv_response = requests.get(csv_url)
    csv_response.raise_for_status()

    # Write the content of the CSV to a file
    filename = os.path.join('data/weekly_market2', link.split('/')[-1])
    with open(filename, 'wb') as file:
        file.write(csv_response.content)
    print(f'Downloaded {filename}')

print('All CSV files from 2023 and above have been downloaded.')
