# timeSeries-processing
Library which processes time series datasets.

The timeseries are daily time series, i.e. the dates are always days.
Are used the [pandas](https://pandas.pydata.org/) built-in types:
    - the dates are represented as pd.Timestamp;
    - vectors of dates are represented as pd.DatetimeIndex;
    - the datasets are represented as pd.DataFrame, indexed by dates (i.e. DatetimeIndex).

There are three groups of functions.
1. Functions to manipulate dates.
2. Function to plot a time series.
3. Processing functions.

This library is part of my bachelor thesis, do check it out the other works.
- [model-selection](https://github.com/EnricoPittini/model-selection) 
- [EEA-datasets-handler](https://github.com/EnricoPittini/EEA-datasets-handler) 
- [ILMETEO-datasets-handler](https://github.com/EnricoPittini/ILMETEO-datasets-handler) 

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install timeSeries-processing.

```bash
pip install timeSeries-processing
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
