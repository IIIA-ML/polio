import json
from datetime import datetime
import pyarrow.parquet as pq
from collections import Counter
import os
import glob


class RetweetDataset:
    def __init__(self, dataset_type, filepaths):
        """
        dataset_type: 'twitter' or 'anonymized'
        filepaths: string (for single file) or list of strings (for multiple parquet files)
        """
        self.dataset_type = dataset_type
        if isinstance(filepaths, str):
            self.filepaths = [filepaths]
        else:
            self.filepaths = filepaths
        self.RTs = []
        self.user_info = {}  # stores username or control info depending on dataset
        self.index_to_accountid = {}
        self.index_to_reposted_postid = {}


    def _parse_ts_twitter(self, ts_str):
        return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%fZ")

    def _json_row_iterator(self, filename):
        with open(filename, 'r') as infile:
            for line in infile:
                row = json.loads(line)
                yield row

    def _parse_twitter(self):
        for file in self.filepaths:
            for row in self._json_row_iterator(file):
                if row.get('referenced_tweets'):
                    if row['referenced_tweets'][0]['type'] == 'retweeted':
                        post_id = row['referenced_tweets'][0]['id']
                        self.RTs.append((
                            row['author_id'],                       # user
                            post_id,                                 # post_id
                            self._parse_ts_twitter(row['created_at'])  # timestamp
                        ))

                        self.user_info[row['author_id']] = row['user']['username']

    def _parquet_row_iterator(self, filename, columns=None):
        parquet_file = pq.ParquetFile(filename)
        for i in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(i, columns=columns)
            for batch in table.to_batches():
                for row in batch.to_pylist():
                    yield row

    def _ids_to_indexes(self):
        account_ids = set()
        reposted_post_ids = set()
        for file in self.filepaths:
            for row in self._parquet_row_iterator(file):
                if not row['is_repost']:
                    continue
                
                account_ids.add(row['accountid'])
                reposted_post_ids.add(row['reposted_postid'])
        
        sorted_account_ids = sorted(account_ids, key=lambda x: str(x))
        sorted_reposted_post_ids = sorted(reposted_post_ids, key=lambda x: str(x))

        # Create a dictionary mapping index -> account_id
        self.index_to_accountid = {idx: account_id for idx, account_id in enumerate(sorted_account_ids)}
        self.index_to_reposted_postid = {idx: post_id for idx, post_id in enumerate(sorted_reposted_post_ids)}

        # Create ID -> index mappings
        accountid_to_index = {account_id: idx for idx, account_id in self.index_to_accountid.items()}
        reposted_postid_to_index = {post_id: idx for idx, post_id in self.index_to_reposted_postid.items()}
        
        return accountid_to_index, reposted_postid_to_index


    def _parse_anonymized(self):
        accountid_to_index, reposted_postid_to_index = self._ids_to_indexes()

        for file in self.filepaths:
            for row in self._parquet_row_iterator(file):
                if not row['is_repost']:
                    continue
                
                accountid = row['accountid']
                reposted_postid = row['reposted_postid']

                idx = accountid_to_index[accountid]
                pidx = reposted_postid_to_index[reposted_postid]

                self.RTs.append((
                    idx,
                    pidx,
                    row['post_time']
                ))
                if accountid not in self.user_info:
                    self.user_info[accountid] = row['is_control']

    def load(self):
        if self.dataset_type == "twitter":
            self._parse_twitter()
        elif self.dataset_type == "anonymized":
            self._parse_anonymized()
        else:
            raise ValueError("Unsupported dataset_type. Use 'twitter' or 'anonymized'.")

        # Number of unique accounts
        unique_accounts = len({acc for acc, _, _ in self.RTs})
        # Number of unique tweets
        unique_tweets = len({tweet for _, tweet, _ in self.RTs})
        print(f"In the dataset there are:\n  {unique_tweets} tweets\n  {unique_accounts} accounts\n  {len(self.RTs)} retweets")

    def filter_min_participation(self, min_participation=2):
        usr_activity = Counter()
        for usr, _, _ in self.RTs:
            usr_activity[usr] += 1

        filtered_RTs = []
        for usr, tweet, time in self.RTs:
            if usr_activity[usr]>=min_participation:
                filtered_RTs.append((usr, tweet, time))



if __name__ == "__main__":
    data = {}
    dataset_names = ['Armenia', 'Bangladesh', 'Catalonia', 'Egypt_UAE', 'Ghana_Nigeria', 'Iran_5', 'Russia_3', 'Spain', 'Venezuela_1', 'Venezuela_2', 'Thailand', 'Ecuador', 'Iran_1', 'Iran_6', 'Russia_5', 'Qatar', 'Russia_2', 'China_1', 'China_2', 'Russia_1', 'Russia_4', 'Iran_2', 'Iran_3', 'Iran_4', 'UAE', 'Cuba']

    for dataset_name in dataset_names:
        processed_dir = f"/data/RTs_data/{dataset_name}/Processed/"
        os.makedirs(processed_dir, exist_ok=True)

        base_dir = f"/data/{dataset_name}/"
        dataset_files = glob.glob(os.path.join(base_dir, "*.gzip.parquet"))
        dataset_files.sort()

        dataset_type = 'anonymized' # Can be 'anonymized' (from the paper Labeled Datasets for Research on Information Operations) 
                                    # or 'twitter'
        dataset_path = dataset_files

        dataset_loader = RetweetDataset(dataset_type, dataset_path)
        dataset_loader.load()

        index_to_accountid_file = f"{processed_dir}index_to_accountid.txt"
        with open(index_to_accountid_file, "w") as f_data:
            f_data.write(f"index_to_accountid={str(dataset_loader.index_to_accountid)}\n")

        file_io_users = f"{processed_dir}/io_users.txt"
        with open(file_io_users, 'w') as f_io:
            for user, is_not_io in dataset_loader.user_info.items():
                if not is_not_io:
                    f_io.write(f"{user}\n")

        file_RTs = f"{processed_dir}RTs.txt"
        with open(file_RTs, 'w') as f_rts:
            f_rts.write(f"RTs={str(dataset_loader.RTs)}")