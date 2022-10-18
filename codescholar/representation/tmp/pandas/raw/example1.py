import pandas as pd

def all_titles():
    global df
    
    allnews = []  # local variable

    for place in ['atalanta', 'bologna']:
        print('search:', place)
        results = get_data_for(place)
        print('found:', len(results))
        allnews += results
        text_download.insert('end', f"search: {place}\nfound: {len(results)}\n")

    df = pd.DataFrame(allnews, columns=['number', 'time', 'place', 'title', 'news', 'link'])
    df = df.sort_values(by=['number', 'time', 'place', 'title'], ascending=[True, False, True, True])
    df = df.reset_index()
                      
    listbox_title.delete('0', 'end')
    
    for index, row in df.iterrows():
        listbox_title.insert('end', row['news'])
