import pandas as pd


def only_thai_numbers(text: str) -> str:
    thai_digits = '๐๑๒๓๔๕๖๗๘๙'
    return ''.join(char for char in text if char in thai_digits)

def thai_to_arabic(thai_number: str) -> str:
    mapping = str.maketrans('๐๑๒๓๔๕๖๗๘๙', '0123456789')
    return thai_number.translate(mapping)

def html_to_dataframe(html: str) -> pd.DataFrame:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    rows = table.find_all('tr')
    
    data = []
    for row in rows:
        cols = row.find_all(['td', 'th'])
        cols = [col.get_text(strip=True) for col in cols]
        data.append(cols)
    
    return pd.DataFrame(data[1:], columns=data[0])

def delete_columns_from_dataframe(df, start_col: int, end_col: int) -> pd.DataFrame:
    return df.drop(df.columns[start_col:end_col], axis=1)       


def apply_to_column(df, col_index: int, func) -> pd.DataFrame:
    df.iloc[:, col_index] = df.iloc[:, col_index].apply(func)
    return df