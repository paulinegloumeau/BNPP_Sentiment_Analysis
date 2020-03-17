The raw CSV file should be put in */data/raw*. This file should at least include the following columns :
- *review_id* the id of the document
- *text* the text of the document

Go to the root of the repo and run `python3 pipeline/run_tokenizer.py`. For each file in */data/raw*, this will create a file in */data/tokenized* with 2 new columns :
- *tokens* all the tokenized words found in the text
- *tokens_index* the index of each token 

Go to the root of the repo and run `python3 pipeline/run_sa.py`. For each file in */data/tokenized*, this will create a file in */data/processed* ending in *_sa* with 1 new column :
- *tokens_sentiment* the sentiment between -1 and 1 of each token

NB : the temporary files are saved with pickle format instead of csv format beacause of some problems we had while saving lists in a dataframe column 