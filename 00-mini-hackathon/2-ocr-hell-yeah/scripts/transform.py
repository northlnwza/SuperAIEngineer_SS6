import pandas as pd


def only_thai_numbers(text: str) -> str:
    thai_digits = '๐๑๒๓๔๕๖๗๘๙'
    return ''.join(char for char in text if char in thai_digits)

def thai_to_arabic_int(thai_number: str) -> int:
    mapping = str.maketrans('๐๑๒๓๔๕๖๗๘๙', '0123456789')
    return int(thai_number.translate(mapping))

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

def delete_null_rows_from_dataframe(df) -> pd.DataFrame:
    return df.dropna()

def apply_to_column(df, col_index: int, func, skipna: bool = True) -> pd.DataFrame:
    df.iloc[:, col_index] = df.iloc[:, col_index].apply(func)
    return df

def convert_column_type(df, col_index: int, new_type: type) -> pd.DataFrame:
    df.iloc[:, col_index] = df.iloc[:, col_index].astype(new_type)
    return df

def update_submission_data(submission_df: pd.DataFrame, extracted_df: pd.DataFrame, doc_id: str) -> pd.DataFrame:
    ocr_map = dict(zip(extracted_df.iloc[:, 0], extracted_df.iloc[:, 1]))
    mask = submission_df['doc_id'] == doc_id
    submission_df.loc[mask, 'votes'] = submission_df.loc[mask, 'party_name'].map(ocr_map).fillna(submission_df.loc[mask, 'votes'])
    
    return submission_df