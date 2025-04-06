import pandas as pd

# create dummy features
def create_dummy_vars(df):
    df['University_Rating'] = df['University_Rating'].astype('object')
    df['Research'] = df['Research'].astype('object')

    clean_data = pd.get_dummies(df, columns=['University_Rating','Research'],dtype='int')

    return clean_data