import requests
from bs4 import BeautifulSoup
import os

# The base URL of the website to scrape
# base_url = "http://reports.ieso.ca/public/RealtimeConstTotals/"
# base_url = 'http://reports.ieso.ca/public/DispUnconsHOEP/'
base_url = 'http://reports.ieso.ca/public/WeeklyMarket/'

# Make a request to the website
response = requests.get(base_url)
response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code

# Parse the HTML content of the page with BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Find all the links on the page that end with '.csv'
csv_links = [link.get('href') for link in soup.find_all('a') if link.get('href').endswith('.csv')]

# Create a directory to store the downloaded files
os.makedirs('data/weekly_market', exist_ok=True)

# Download each CSV file
for link in csv_links:
    csv_url = base_url + link
    csv_response = requests.get(csv_url)
    csv_response.raise_for_status()

    # Write the content of the CSV to a file
    filename = os.path.join('data/weekly_market', link.split('/')[-1])
    with open(filename, 'wb') as file:
        file.write(csv_response.content)
    print(f'Downloaded {filename}')

print('All CSV files have been downloaded.')