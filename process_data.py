

from Main.sort import sort_dataset

if __name__ == "__main__":
    for dataset in ['Weibo', 'DRWeiboV3', 'Twitter15-tfidf', 'Twitter16-tfidf']:
        src_path = f"data/{dataset}/source"
        dataset_path = f"data/{dataset}/dataset"
        sort_dataset(src_path, dataset_path)