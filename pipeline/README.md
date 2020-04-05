# Aspect based sentiment analysis pipeline

The goal of this project was to build a complete pipeline, taking a set of documents as an input and processing it to give insights on the different topics discussed and the satisfaction associated.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You will need Python 3 to run the pipeline, we highly recommend working in a conda environment. 

This project was mainly made using windows, you may need to change some code if you get path errors. 

### Installing

Go to the root of the project and install the python modules : 

```
pip install -r pipeline/requirements.txt
```

You will also want to make sure you have pytorch installed. If you are unsing a conda env, we recommend : 

```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

## Using the pipeline

### Data

We tested our pipeline with a part of the Yelp Dataset available on Kaggle : (https://www.kaggle.com/yelp-dataset/yelp-dataset). This dataset provides reviews made by users about many businesses (restaurants, hair salons, ...). 


The [input example](data/raw/input_example.csv) we provide can be used to make sure the pipeline works. It is a subset of the Yelp Dataset, made of 100 reviews dealing with Auto Repairs. 

Any input should be a csv with at least :
- a `review_id` column
- a `text` column 


### Preprocessing

Run 

```
python pipelin/run_preprocesser.py
```

For each in put file in `data/raw`, a pickle file (with `_preprocessed` added to its original name) will be created and saved in `data/preprocessed`.

### Aspect Detection

Run 

```
python pipelin/run_aspect_detection_louvain.py
```

For each in put file in `data/preprocessed`, a pickle file (with `_ad` added to its original name) will be created and saved in `data/processed`.

### Sentiment Analysis

Run 

```
python pipelin/run_sa.py
```

For each in put file in `data/preprocessed`, a pickle file (with `_sa` added to its original name) will be created and saved in `data/processed`.

### Putting it all together

Run 

```
python pipelin/run_gathering.py
```


## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc





The raw CSV file should be put in */data/raw*. This file should at least include the following columns :
- *review_id* the id of the document
- *text* the text of the document

Go to the root of the repo and run `python3 pipeline/run_preprocesser.py`. For each file in */data/raw*, this will create a file in */data/tokenized* with 2 new columns :
- *tokens* all the tokenized words found in the text
- *tokens_index* the index of each token 

Go to the root of the repo and run `python3 pipeline/run_sa.py`. For each file in */data/tokenized*, this will create a file in */data/processed* ending in *_sa* with 1 new column :
- *tokens_sentiment* the sentiment between -1 and 1 of each token

NB : the temporary files are saved with pickle format instead of csv format beacause of some problems we had while saving lists in a dataframe column 





 nano C:\Users\Arthur\Anaconda3\envs\py36\lib\site-packages\apex\reparameterization\weight_norm.py retirer fused weight norm