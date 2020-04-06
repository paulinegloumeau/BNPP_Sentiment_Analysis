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

Based on [NVIDIA's sentiment neuron](https://github.com/NVIDIA/sentiment-discovery).

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

### Data Visualization

Run 
```
python .\pipeline\visualize.py --heatmap --global-analysis
```

## Authors

* **Pauline Gloumeau**
* **Arthur Goutallier**
* **Victor Le Fourn**

## License

This project is licensed under the MIT License - see the [LICENSE.md](../LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc





