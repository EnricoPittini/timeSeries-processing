# timeSeries-processing
Library which processes time series datasets.

This library is part of my bachelor thesis, check out the other works.
- [model-selection](https://github.com/EnricoPittini/model-selection)
- [EEA-datasets-handler](https://github.com/EnricoPittini/EEA-datasets-handler)
- [ILMETEO-datasets-handler](https://github.com/EnricoPittini/ILMETEO-datasets-handler)
- [Air-quality-prediction](https://github.com/EnricoPittini/Air-quality-prediction)

## Description
The supported time series are daily time series, which means that the dates are always days.

### Purpose
This library is a tool for time series modeling. In particular, it is an auxiliary utility for helping building machine learning models for time series forecasting.

In fact, the main application of this library is to, given a time series dataset, add some useful and interesting time-related features to it.
In other words, it allows the user to extract and build some important time-related explanatory features.

These new features are obtained from a specific and already existing feature of the dataset, by selecting, grouping and processing the days which are somehow related to the ones in the given dataset.
As a result, each of these new computed features is an indicator of the behaviour of the specific feature but in other related days.

For example, given a time series dataset and specifying a certain feature, it is possible to add some new features representing the specified feature but in the previous days.
Each new feature indicates the value of the specified feature in a certain previous day.

The interfaces of the functionalities of the library are simple and intuitive, but they are also rich. In this way, the user is able to personalize the time series operations in a powerful and flexible way.

### Functionalities
There are three groups of functionalities.

The first group is able to manipulate dates (i.e. days). There are several different operations.
For example, one of them is able to split a collection of days by a certain criterion, which can either be year, month or season.
These functionalities are mainly built in order to be some auxiliary utilities for the other functionalities.

The second group is able to plot time series values.
The user can specify several different options, in order to change the visualization and the division of the values. This can be particularly useful for understanding some time-related patterns, like seasonal behaviours.

The third group of functionalities is the most important. These are the processing functionalities, i.e. the ones which actually process the time series datasets.
As described above, the main purpose of these functionalities is to extract and build interesting time-related explanatory features.

### Implementation details
This library is built on top of the pandas library.
The pandas built-in data types are indeed used.
- The dates are represented with the pd.Timestamp type.
- Vectors of dates are represented with the pd.DatetimeIndex type.
- The time series datasets are represented as pd.DataFrame indexed by dates (i.e. the index is a  pd.DatetimeIndex).
In addition, several pandas utilities and methods are used.

Each processing functionality of timeSeries-processing adds the new extracted features to the given dataset by producing a new dataset, i.e. the given dataset is not modified.
In addition, each processing functionality also returns two NumPy arrays: the first is X, which contains the explanatory features of the returned dataset; the second is y, which contains the response feature of the returned dataset.
In other words, each of these functionalities automatically splits the obtained dataset into the features used to make the predictions and the feature which is the target of the prediction.
This can be particularly useful to easily build and evaluate different machine learning models in a compact way.

To conclude, the time series plotting is built on top of the Matplotlib library.

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install timeSeries-processing.

```bash
pip install timeSeries-processing
```

## References
- [matplotlib](https://matplotlib.org/stable/index.html) is a comprehensive library for creating static, animated, and interactive visualizations in Python.
- [pandas](https://pandas.pydata.org/) is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool,
built on top of the Python programming language.
- [sklearn](https://scikit-learn.org/stable/index.html), machine Learning in Python.

## License
[MIT](https://choosealicense.com/licenses/mit/)
