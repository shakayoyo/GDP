import os 
import pandas as pd
from utils import set_seed, path_expander

#shoppe dataset

def load_shopee_data(base_dir: str):
    train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
    test_df  = pd.read_csv(os.path.join(base_dir, 'test.csv'))
    train_df['image'] = train_df['image'].apply(lambda p: path_expander(p, base_dir))
    test_df['image']  = test_df['image'].apply(lambda p: path_expander(p, base_dir))
    return train_df, test_df

## Yelp dataset

def load_review_dataset(
    csv_path: str,
    seed: int,
    test_size: float = 0.2
):
    """
    Load the Yelp review dataset from CSV, split target reviews into train/test,
    and return source reviews.

    Parameters:
      - csv_path: path to the 'simple_review.csv' file
      - seed: random seed for reproducibility
      - test_size: fraction of target data for test split

    Returns:
      train_df: DataFrame of target reviews for training (columns ['stars','text'])
      test_df : DataFrame of target reviews for testing  (columns ['stars','text'])
      source_df: DataFrame of source reviews (cool == 0) (columns ['stars','text'])
    """
    # Read full DataFrame
    df = pd.read_csv(csv_path)

    # Partition into target (cool != 0) and source (cool == 0)
    target_df = df[df['cool'] != 0][['stars', 'text']].copy()
    source_df = df[df['cool'] == 0][['stars', 'text']].copy()

    # Ensure reproducibility
    set_seed(seed)

    # Split target into train and test
    train_df, test_df = train_test_split(
        target_df,
        test_size=test_size,
        random_state=seed,
        shuffle=True
    )

    return train_df, test_df, source_df

