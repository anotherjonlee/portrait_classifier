
def img_downloader():
    
    """
    Upon execution the function will access the image source and download images and their associated
    metadata. The function will generate a backup json file as well as out put a list of dictionary
    that can be converted into a pandas dataframe.
    """
    
    from bs4 import BeautifulSoup
    import requests
    import time
    import json
    import os
    import sys
    sys.path.append("..")

    base_url = 'http://numismatics.org/search/results?q=department_facet%3A%22Roman%22%20AND%20year_num%3A%5B-30%20TO%20%2A%5D%20AND%20imagesavailable%3Atrue&lang=en&layout=grid&start='
        
    iter_count = 0
    dict_list = []

    if not os.path.exists('../data/raw_imgs/'):
        os.makedirs('../data/raw_imgs/')
        
    print('Downloading target images and parsing their metadata.')    
    
    # Web scraping a list of all image URLs
    for i in range(20, 45400, 20):
        error_counter = 0

        iter_count += 1
        
        if i % 200 == 0:
            print(f"sleeping for 2 seconds after {i}'th iteration.")
            time.sleep(2)

        url = base_url + str(i)

        code = requests.get(url).status_code

        if code == 200:

            try:

                html_content = requests.get(url).text
                soup = BeautifulSoup(html_content)
                thumbnail_urls = soup.find_all('a',{"class": "thumbImage"})

                for tn_url in range(0,len(thumbnail_urls),2):

                    small_img_url = thumbnail_urls[tn_url].find('img')['src']
                    metadata_url = thumbnail_urls[tn_url]['id']

                    metadata_content = requests.get(metadata_url).text
                    metadata_soup = BeautifulSoup(metadata_content)
                    metadata = metadata_soup.find_all('div',{'class': 'metadata_section'})

                    all_lis = metadata[1].find_all('li')

                    keys = ['authority','portrait','year','obj','denomination','material','region','deity']

                    temp_dict = {key:'NA' for key in keys}

                    for li in all_lis:
                        if li.find('b').get_text() == 'Authority: ':
                            temp_dict['authority'] = li.find('a').contents[0].replace('/','_')
                        elif li.find('b').get_text() == 'Portrait: ':
                            temp_dict['portrait'] = li.find('a').contents[0].replace('/','_')
                        elif li.find('b').get_text() == 'From Date: ':
                            temp_dict['year'] = li.find('span').contents[0]
                        elif li.find('b').get_text() == 'Object Type: ':
                            temp_dict['obj'] = li.find('a').contents[0]
                        elif li.find('b').get_text() == 'Denomination: ':
                            temp_dict['denomination'] = li.find('a').contents[0]
                        elif li.find('b').get_text() == 'Material: ':
                            temp_dict['material'] = li.find('a').contents[0]
                        elif li.find('b').get_text() == 'Region: ':
                            temp_dict['region'] = li.find('a').contents[0]

                        elif li.find('b').get_text() == 'Deity: ':
                            temp_dict['deity'] = li.find('a').contents[0]


                    temp_dict['url'] = small_img_url

                    img_r = requests.get(small_img_url)

                    fname = f"{temp_dict['authority']}_{temp_dict['portrait']}_{time.time()}"

                    temp_dict['fname'] = fname

                    open(f'imgs/{fname}.jpg' , 'wb').write(img_r.content)

                    dict_list.append(temp_dict)

            except:
                print(f'error encountered at URL: {requests.get(url).status_code}')
                print(f'error encountered at metadata URL: {requests.get(metadata_url).status_code}')
                print(f'i: {i}')
                print(f'k: {tn_url}')
                pass
            
        else:
            error_counter += 1
            time.sleep(10)
            pass

        if error_counter == 10:
            print(f"Stopped downloading images because the function encountered 10 errors.'")
            print(f'page number where a 10th error was encountered: {i}')
            break

    with open('../data/raw_metadata.json', 'w') as json_f:
        json.dump(dict_list, json_f)

    print('Downloading and parsing processes completed.')
    
    return dict_list
