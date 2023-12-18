import requests
from bs4 import BeautifulSoup

def web(url):
    response=requests.get(url)
    if response.status_code==200:
        soup=BeautifulSoup(response.content,'html.parser')
        print(soup.title.strings)
        print(soup.get_text())
    else:
        print(response.status_code)
urls="https://github.com/Godwin02"
web(urls)