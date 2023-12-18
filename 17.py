import  requests
def web(url):
    response=requests.get(url)
    if response.status_code==200:
        print(response.text)
    else:
        print("Not available",response.status_code)
urls="https://github.com/Godwin02"
web(urls)
