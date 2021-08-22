"""
Module which processes time series datasets.

These time series are daily time series, i.e. the dates are always days. (Dates and days are used as synonyms).
Are used the pandas built-in types:
    - The dates are represented as pd.Timestamp.
    - Vectors of dates are represented as pd.DatetimeIndex.
    - The datasets are represented as pd.DataFrame, indexed by dates (i.e. the index is a pd.DatetimeIndex).

There are three groups of functions.
    1. Functions to manipulate dates.
    2. Function to plot a time series.
    3. Processing functions.
"""

import warnings

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter("always")




#----------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS TO MANIPULATE DATES


def find_missing_days(days):
    """
    Return, given a vector of days, his missing days.

    More specifically, the missing days are the ones which are not present in the contiguous sequence of days in `days`.

    Parameters
    ----------
    days: pd.DatetimeIndex
        Vector of dates.

    Returns
    ----------
    pd.DatetimeIndex
        Vector of missing days.
    """
    day_min = min(days)
    day_max = max(days)

    return pd.date_range(start=day_min,end=day_max).difference(days)


def find_same_month_days(day):
    """
    Return, given a day, all the days which are in the same month.

    Parameters
    ----------
    day: pd.Timestamp

    Returns
    ----------
    pd.DatetimeIndex
        Vector of the days in the same month.
    """
    first_day = day - pd.DateOffset(day.day - 1)
    last_day = first_day + pd.DateOffset(day.days_in_month - 1)
    return pd.date_range(start=first_day,end=last_day)


def find_same_season_days(day):
    """
    Return, given a day, all the days which are in the same season.

    The meteorological seasons are considered, and not the astronomical ones.

    Parameters
    ----------
    day: pd.Timestamp

    Returns
    ----------
    pd.DatetimeIndex
        Vector of the days in the same season.
    """
    year = day.year

    seasons_end_days = [(pd.Timestamp(year-1,12,1),pd.Timestamp(year,2,29 if year%4==0 else 28)),
                   (pd.Timestamp(year,3,1),pd.Timestamp(year,5,31)),
                   (pd.Timestamp(year,6,1),pd.Timestamp(year,8,31)),
                   (pd.Timestamp(year,9,1),pd.Timestamp(year,11,30)),
                   (pd.Timestamp(year,12,1),pd.Timestamp(year+1,2,29 if (year+1)%4==0 else 28))]

    for start,end in seasons_end_days:
        if day>=start and day<=end:
            return pd.date_range(start,end)


def find_k_years_ago_days(day, k=1, n_days=11):
    """
    Return, given a day, the days which are centered on that day but k years ago.

    Parameters
    ----------
    day: pd.Timestamp
    k: int
        Indicates which past year has to be considered (i.e. `k` years ago).
    n_days: int or str
        Indicates specifically which are the `k` years ago to select.
            - If it's an int, it must be an odd positive number. The `n_days` centered on `day` but `k` years
              ago are selected .
            - If it's a str, it must be either "month" or "season". All the days in the same month/season but
              `k` years ago are selected.
              (The meteorological seasons are considered, and not the astronomical ones)

    Returns
    ----------
    pd.DatetimeIndex
        Vector of the selected days.

    Raises
    ----------
    ValueError
        When `n_days` is neither an odd positive integer nor "month" nor "season".
    """
    year = day.year
    month = day.month
    d = day.day
    if month==2 and d==29 and (year-k)%4!=0: # Day that is not present k years ago (29 of February)
        k_years_ago_day = pd.Timestamp(year-k,month,28)
    else:
        k_years_ago_day = pd.Timestamp(year-k,month,d)

    if type(n_days)==int:
        if n_days%2==0:
            raise ValueError("n_days must be either an odd integer or \"month\" or \"season\"")
        first_day = k_years_ago_day - pd.DateOffset(n_days//2)
        res_days = pd.date_range(first_day,first_day+pd.DateOffset(n_days-1))

    elif n_days == "month":
        res_days = find_same_month_days(k_years_ago_day)

    elif n_days == "season":
        res_days = find_same_season_days(k_years_ago_day)

    else:
        raise ValueError("n_days must be either an odd integer or \"month\" or \"season\"")

    return res_days


def find_current_year_days(day, n_days=11, current_day=False):
    """
    Return, given a day, the preceding days of the same year which are centered on that day.

    Parameters
    ----------
    day: pd.Timestamp
    n_days: int or str
        Indicates specifically which are the current year days to select.
            - If it's an int, the `n_days` preceding `day` are selected.
            - If it's a str, it must be either "month" or "season". All the days in the same month/season that
              precede `day` are selected.
              (The meteorological seasons are considered, and not the astronomical ones)
    current_day: bool
        Indicates whether to select also the current day (i.e. `day`) or not.

    Returns
    ----------
    pd.DatetimeIndex
        Vector of the selected days.

    Raises
    ----------
    ValueError
        When `n_days` is neither an integer nor "month" nor "season".
    """

    if type(n_days)==int:
        first_day = (day - pd.DateOffset(n_days-1)) if current_day else (day - pd.DateOffset(n_days))
        res_days = pd.date_range(first_day,first_day+pd.DateOffset(n_days-1))

    elif n_days == "month":
        res_days = find_same_month_days(day)
        if current_day:
            res_days = res_days[res_days<=day]
        else:
            res_days = res_days[res_days<day]

    elif n_days == "season":
        res_days = find_same_season_days(day)
        if current_day:
            res_days = res_days[res_days<=day]
        else:
            res_days = res_days[res_days<day]

    else:
        raise ValueError("n_days must be either an integer or \"month\" or \"season\"")

    return res_days


def group_days_by(days, criterion):
    """
    Group the given vector of days according to the given criterion.

    Parameters
    ----------
    days: pd.DatetimeIndex
    criterion: str
        Indicates how to group the given days. It can be either "year" or "month" or "season".
        (The meteorological seasons are considered, and not the astronomical ones)

    Returns
    ----------
    list
        List of pairs (i.e. tuples).
        Each pair is a group of days.
            - The first element is a string which represents the group name (i.e. group label).
            - The second element is the vector of days in that group, i.e. it's a pd.DatetimeIndex.

    Raises
    ----------
    ValueError
        When `criterion` is neither "year" nor "month" nor "season".

    Notes
    ----------
    For the sake of completeness, it's important to say that if `criterion` is either "month" or "season", also days of
    different years could be grouped together.
    """
    days = days.sort_values()
    if criterion=="year":
        years = days.year.drop_duplicates()
        return [(str(year),days[days.year==year]) for year in years]
    elif criterion=="month":
        def stringify_month(month):
            return ["January","February","March","April","May","June","July","August","September",
                    "October","November","December"][month-1]
        months = days.month.drop_duplicates()
        return [(stringify_month(month),days[days.month==month]) for month in months]
    elif criterion=="season":
        def to_season(month):
            return ["Winter", "Spring", "Summer", "Fall"][month%12//3]
        seasons = days.month.map(lambda month: to_season(month)).drop_duplicates()
        return [(season,days[list(days.month.map(lambda month: to_season(month)==season))]) for season in seasons]
    else:
        raise ValueError("criterion must be either \"year\" or \"month\" or \"season\" ")




#----------------------------------------------------------------------------------------------------------------------------
# FUNCTION TO PLOT A TIME SERIES


def plot_timeSeries(df, col_name, divide=None, xlabel="Days", line=True, title="Time series values", figsize=(9,9)):
    """
    Plot a column of the given time series DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame indexed by days (i.e. the index is a pd.DatetimeIndex).
    col_name: str
        Indicates the specified column to plot.
    divide: str
        Indicates if and how to divide the plotted values.
        It can either be None, "year", "month" or "season". (The meteorological seasons are considered, and not the
        astronomical ones).
        That division is simply made graphically using different colors.
    xlabel: str
        Label to put on the x axis.
    line: bool
        Indicates whether to connect the points with a line.
    title: str
        Title of the plot.
    figsize: tuple
        Dimensions of the plot.

    Returns
    ----------
    matplotlib.axes.Axes
        The matplotlib Axes where the plot has been made.
    """

    fig, ax = plt.subplots(figsize=figsize)

    if not divide:
        ax.plot(df.index, df[col_name], 'o:' if line else 'o')
    else:
        groups = group_days_by(df.index, criterion=divide)
        color = None
        for group in groups:
            if divide=="season":
                colors = {"Winter":"blue", "Spring":"green", "Summer":"yellow", "Fall":"red"}
                color = colors[group[0]]
            elif divide=="month":
                colors = {"January":"b",
                          "February":"g",
                          "March":"r",
                          "April":"c",
                          "May":"m",
                          "June":"y",
                          "July":"k",
                          "August":"peru",
                          "September":"crimson",
                          "October":"orange",
                          "November":"darkgreen",
                          "December":"olivedrab"}
                color = colors[group[0]]
            ax.plot(group[1], df.loc[group[1],col_name], 'o:' if line else 'o', color=color , label=group[0])


    ax.set_xlabel(xlabel)
    ax.set_ylabel(col_name)
    ax.set_title(title)
    ax.grid()
    if divide:
        ax.legend()

    return ax




#----------------------------------------------------------------------------------------------------------------------------
# PROCESSING FUNCTIONS


def split_X_y(df, y_col=None, scale_y=True):
    """
    Split the given DataFrame into X and y.

    X is a matrix which contains the explanatory variables of `df`, y is a vector which contains the response variable of
    `df` (i.e. the variable which is the target of the prediction analysis tasks).
    Optionally, the values in y can be scaled.

    This function is an auxiliary utility for the processing functions.

    Parameters
    ----------
    df: pd.DataFrame
    y_col: str
        Indicates which is the `df` column that is the response feature.
        If it is None, the last `df` column is considered.
    scale_y: bool
        Indicates whether to scale or not the values in y.

    Returns
    ----------
    X: np.array
        Two-dimensional np.array, containing the explanatory features of `df`.
    y: np.array
        Mono dimensional np.array, containing the response feature of `df`.

    Notes
    ----------
    The scaling of the values in y is performed using the sklearn MinMaxScaler.
    """
    if y_col is None:
        y_col = df.columns[-1]

    y = df[y_col].values # Numpy vector y
    X = df.drop([y_col],axis=1).values # Numpy matrix X

    if scale_y: # Scale the y
        scaler= MinMaxScaler()
        scaler.fit(y.reshape(y.shape[0],1))
        y = scaler.transform(y.reshape(y.shape[0],1)).reshape(y.shape[0],)

    return X,y


def add_timeSeries_dataframe(df, df_other, y_col=None, scale_y=True):
    """
    Add to a time series DataFrame another time series DataFrame.

    The two DataFrames are concatenated into a new DataFrame, i.e. the resulting DataFrame contains all the columns in `df`
    and `df_other`.
    This concatenation is done with respect to the former DataFrame: this means that all the days of `df` are kept, while
    only the days of `df_other` that are also in `df` are kept.

    In addition, the resulting DataFrame is automatically  split into the X matrix and the y vector, which are respectively
    the matrix containing the explanatory features and the vector containing the response feature.
    (The response feature is the one which is the target of the prediction analysis tasks).

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame indexed by days (i.e. the index is a pd.DatetimeIndex).
    df_other: pd.DataFrame
        Other DataFrame indexed by days (i.e. the index is a pd.DatetimeIndex), which has to be added to the former.
    y_col: str
        Indicates which is the column of the resulting DataFrame to be used as y column.
    scale_y: bool
        Indicates whether to scale or not the values of the response feature y.

    Returns
    ----------
    pd.DataFrame
        The DataFrame resulting from the concatenation.
    X: np.array
        Two-dimensional numpy array which contains the explanatory features.
    y: np.array
        Mono-dimensional numpy array which contains the response feature.

    See Also
    ----------
    split_X_y: splits a DataFrame into X and y.
    """
    df_other = df_other.loc[df.index]
    df =  pd.concat([df,df_other],axis=1)

    X,y = split_X_y(df,y_col,scale_y=scale_y)

    return df,X,y


def add_k_previous_days(df, col_name, k, y_col=None, scale_y=True):
    """
    Add, to a time series DataFrame, features containing values of the specified column but related to the previous days.

    A new DataFrame is built, which is created from `df` adding `k` new columns. These `k` new columns contain the values
    of the column `col_name` but with regard to, respectively: the day before; the 2-days before; ... ; the k-days before.
    In this way, in the resulting DataFrame for each day there is information about the feature `col_name` up to `k` days
    before.
    These `k` columns are, respectively, called: "col_name_1", "col_name_2", ..., "col_name_k".

    The first `k` days are removed from the resulting DataFrame: that is because for the first `k` days there isn't enough
    information to build the new `k` columns.

    In addition, the resulting DataFrame is automatically  split into the X matrix and the y vector, which are respectively
    the matrix containing the explanatory features and the vector containing the response feature.
    (The response feature is the one which is the  target of the prediction analysis tasks).

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame indexed by days (i.e. the index is a pd.DatetimeIndex).
    col_name: str
        Indicates which is the column to be used to build the `k` new columns.
    k: int
        Indicates how many previous days are to be taken into account (i.e. how many new columns are built).
    y_col: str
        Indicates which is the column of the resulting DataFrame to be used as y column.
    scale_y: bool
        Indicates whether to scale or not the values of the response feature y.

    Returns
    ----------
    pd.DataFrame
        The resulting DataFrame.
    X: np.array
        Two-dimensional numpy array which contains the explanatory features.
    y: np.array
        Mono-dimensional numpy array which contains the response feature.

    Raises
    ----------
    ValueError
        When `df` does not contain a contiguous sequence of days (i.e. there are missing days in `df`).

    See Also
    ----------
    split_X_y: splits a DataFrame into X and y.
    """

    if len(find_missing_days(df.index))>0: # In df there isn't a contigous sequence of days
        raise ValueError("In df there must be a contigous sequence of days")

    # Copy the DataFrame, removing the first k rows. df_new will be the resulting DataFrame.
    df_new = df.copy().iloc[k:]

    # Iterate from 1 to k: at each iteration is made a new column (i.e. at the iteration "i" it's made the i-th new column).
    # At the itearation "i", the considered day is the one which is "i" days before the current day (i.e. "i"-th previous
    # day).
    for i in range(1,k+1):
        # DataFrame with only one column, containing all the values of the column col_name with respect to the "i"-th
        # previous day.
        # Basically, this is the "i"-th new column to add to the DataFrame.
        df_to_add = df[[col_name]].copy().iloc[k-i:-(i)]
        df_to_add = df_to_add.rename(columns={col_name:col_name+"_"+str(i)}) # Rename the column
        df_to_add = df_to_add.set_index(df_new.index) # Set as index the actual days (i.e. index of df)

        df_new = pd.concat([df_new,df_to_add],axis=1) # Add the new column

    X,y = split_X_y(df_new, y_col, scale_y=scale_y)

    return df_new,X,y


def add_k_years_ago_statistics(df, df_k_years_ago, k=1, days_to_select=11, stat="mean", columns_to_select=None,
                               replace_miss=True, y_col=None, scale_y=True):
    """
    Add, to a time series DataFrame, statistics computed on the other given time series DataFrame, but with respect to the
    days of k years ago.

    `df_k_years_ago` should contain days of `k` years ago with respect to the days of `df`.(Nevertheless, both `df` and
    `df_k_years_ago` can contain multiple years).
    Let 'm' be the number of the selected columns of `df_k_years_ago` (by default all the columns, see the
    `columns_to_select` parameter).
    A new DataFrame is built, which is created from `df` adding `m` new columns. (The resulting DataFrame has the same
    index of `df`). These new `m` columns contain the values computed from the associated columns of `df_k_years_ago`
    considering the days of `k` years before the ones in `df`.

    Going into the details, let 'day' be a row of `df`, and 'new_column' be one of the 'm' new columns created
    in the resulting DataFrame. The value put in that column for that day is computed from the associated column of
    `df_k_years_ago` considering the days of `df_k_years_ago` that are centered on `day` but k years ago.
    (See the find_k_years_ago_days function).
    Once the `k` years ago days in `df_k_years_ago` are selected, an unique value for the new column 'new_column' and for
    the day 'day' is computed applying a certain statistical aggregation (specified by the input parameter `stat`) on the
    values of these selected days in the column of `df_k_years_ago` associated to 'new_column'.

    `days_to_select` specifies, for each 'day' of `df`, which `k` years ago days in `df_k_years_ago` are selected. The
    semantics is quite similar to the parameter `n_days` of the find_k_years_ago_days function (it can be either an odd
    integer or "month" or "season").

    Actually, `days_to_select` can also be more powerful than that. `days_to_select` can be a predicate (i.e a function that
    returns a bool), which is used to select the days of `k` years ago: for each 'day' of `df`, the `k` years ago days
    in `df_k_years_ago` for which the function `days_to_select` returns True are selected.
    So, `days_to_select` is a predicate that, in a flexible way, selects the days of `k` years ago.
    The signature of the function must be: (day: pd.TimeStamp, df: pd.DataFrame, k_years_ago_day: pd.TimeStamp,
    df_k_years_ago: pd.DataFrame): bool.
    Where:
        - `day` is the current day of `df` ;
        - `df` is the given DataFrame ;
        - `day_k_years_ago` is the day of `k` years ago contained in `df_k_years_ago`;
        - `df_k_years_ago` is the other given DataFrame, containing days of `k` years ago.
    The function returns True if and only if `day_k_years_ago` is a day that has to be selected for `day`.

    For a certain 'day' of `df` it could happen that no `k` years ago day is selected. This means that this 'day' has a
    missing value for each of the 'm' new columns (i.e. 'm' missing values).
    In this case, if `replace_miss` is True, all the missing values are filled: the missing value for the new column
    'new_column' is filled computing the mean of all the values in the associated column in `df_k_years_ago`.
    Otherwise, if `replace_miss` is False, the 'm' missing values are simply kept as Nan.

    So, in the end, 'm' new columns are created in the resulting DataFrame, from the selected 'm' columns of
    `df_k_years_ago`.
    From the selected column with name "col" of `df_k_years_ago`, the corresponding column "k_years_ago_col" is created in
    the resulting DataFrame.

    In addition, the resulting DataFrame is automatically  split into the X matrix and the y vector, which are respectively
    the matrix containing the explanatory features and the vector containing the response feature.
    (The response feature is the one which is the target of the prediction analysis tasks).

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame indexed by days (i.e. the index is a pd.DatetimeIndex).
    df_k_years_ago: pd.DataFrame
        DataFrame indexed by days (i.e. the index is a pd.DatetimeIndex). It should contain days of `k` years ago with
        respect to the days in `df`.
    k: int
        Indicates which previous year, with respect to the days in `df`, has to be taken into account (i.e. the year which is
        `k` years ago).
        It must be a positive integer.
    days_to_select: int or str or callable
        Indicates, for each day of `df`, which `k` years ago days are selected in `df_k_years_ago`.
        It must either be an odd integer or "month" or "season" or a predicate (i.e. a function that returns a boolean).
        The function signature must be
                (day: pd.TimeStamp, df: pd.DataFrame, day_k_years_ago: pd.TimeStamp, df_k_years_ago: pd.DataFrame): bool
    stat: str
        Indicates the statistical aggregation to perform, for each day of `df`, on the selected `k` years ago days of
        `df_k_years_ago`.
        It can either be "mean" or "min" or "max".
    columns_to_select: list
        List of strings which indicates the columns of `df_k_years_ago` that have to be taken into account.
        If it's None, all the columns of `df_k_years_ago` are considered.
    replace_miss: bool
        Indicates whether to fill the missing values or keep them as Nan.
        (The missing values are generated for each day of `df` for which no `k` years ago day in `df_k_years_ago` is
        selected).
    y_col: str
        Indicates which is the column of the resulting DataFrame to be used as y column.
    scale_y: bool
        Indicates whether to scale or not the values of the response feature y.

    Returns
    ----------
    pd.DataFrame
        The resulting DataFrame.
    X: np.array
        Two-dimensional numpy array which contains the explanatory features.
    y: np.array
        Mono-dimensional numpy array which contains the response feature.

    Raises
    ----------
    ValueError
        - When `k` is not a positive integer.
        - When `stat` is neither "mean" nor "min" nor "max".

    Warns
    ----------
    UserWarning
        - When, for a day of `df`, no `k` years ago day in `df_k_years_ago` is selected.
        - When, for a day of `df`, less `k` years ago days are found compared to the ones expected.
          (This can happen only if `days_to_select` is either an odd integer or "month" or "season").

    See Also
    ----------
    find_k_years_ago_days: returns, given a day, the selected days of k years ago.
    split_X_y: splits a DataFrame into X and y.

    Notes
    ----------
    If add_k_years_ago_statistics is applied multiple times with the same `k` on the same `df` and `df_k_years_ago`,
    columns with the same name are potentially created.
    For instance, if add_k_years_ago_statistics is applied three times with the same `k` on the same DataFrames, from the
    `df_k_years_ago` column "col" three different columns with the same name "k_years_ago_col" are potentially created.
    To avoid that, add_k_years_ago_statistics ensures that all the different columns with the same name are properly
    disambiguated, using progressive numbers. (E.g three different columns with the same name "k_years_ago_col", became
    "k_years_ago_col", "k_years_ago_col.1" and "k_years_ago_col.2").
    """

    if k<=0:
        raise ValueError("k must be a positive number")

    df = df.copy()

    if not columns_to_select: # If columns_to_select is None, all the `df_k_years_ago` columns are taken into account
        columns_to_select = list(df_k_years_ago.columns)

    # Two dimensional list (matrix-like). It contains, for each day of `df`, the values computed on `df_k_years_ago` for that
    # day (i.e. the values to be added in the resulting DataFrame).
    # So, it's a matrix that contains as many rows as the days in `df` and as many columns as the selected columns in
    # `df_k_years_ago`.
    statistics_list = []

    for day in df.index: # Iterate through all the days in `df`

         if not callable(days_to_select): # days_to_select is not a function : either an odd integer or "month" or "season"
             k_years_ago_days = find_k_years_ago_days(day, k, days_to_select) # Selected `k` years ago days (expected)
             expected_n_days = len(k_years_ago_days) # Number of these expected `k` years ago days
             # Actual `k` years ago days in `df_k_years_ago`
             k_years_ago_days = k_years_ago_days[list(k_years_ago_days.map(lambda day_k_years_ago: (day_k_years_ago in
                                                                                                    df_k_years_ago.index) ))]

         else: # days_to_select is a function
             # Selected `k` years ago days (The ones filtered by the function)
             k_years_ago_days = df_k_years_ago.index[
                                                    list(df_k_years_ago.index.map(lambda day_k_years_ago: \
                                                                                        (day_k_years_ago.year+k==day.year and
                                                                                        days_to_select(day, df,
                                                                                                         day_k_years_ago,
                                                                                                         df_k_years_ago)
                                                                                        )
                                                                                  )
                                                        )
                                                    ]

         if len(k_years_ago_days)==0: # No `k` years ago day has been selected
             warnings.warn("No "+str(k)+" years ago days have been found for the day " + day.strftime('%Y-%m-%d'))
         # Less `k` years ago days have been selected than the ones expected
         elif not callable(days_to_select) and len(k_years_ago_days)<expected_n_days:
             warnings.warn(("For the day "+day.strftime('%Y-%m-%d')+
                            " only these "+str(k)+" years ago days have been found: "+
                            str(list(k_years_ago_days.strftime('%Y-%m-%d')))))

         # No `k` years ago day has been selected and `replace_miss` is True: the missing values are filles with the
         # means of the columns.
         # (If `replace_miss`, is False the missing values are kept as Nan)
         if len(k_years_ago_days)==0 and replace_miss:
            if stat=="mean":
                statistics_list.append([df_k_years_ago[column].mean() for column in columns_to_select ])
            elif stat=="min":
                statistics_list.append([df_k_years_ago[column].min() for column in columns_to_select ])
            elif stat=="max":
                statistics_list.append([df_k_years_ago[column].max() for column in columns_to_select ])
            else:
                raise ValueError("stat must be a statistical measure : \"mean\" or \"min\" or \"max\" ")
         else: # There is at least one selected `k` years ago day (or `replace_miss` is False)
            if stat=="mean":
                statistics_list.append(list(df_k_years_ago[columns_to_select].loc[k_years_ago_days].mean()))
            elif stat=="min":
                statistics_list.append(list(df_k_years_ago[columns_to_select].loc[k_years_ago_days].min()))
            elif stat=="max":
                statistics_list.append(list(df_k_years_ago[columns_to_select].loc[k_years_ago_days].max()))
            else:
                raise ValueError("stat must be a statistical measure : \"mean\" or \"min\" or \"max\" ")

    # Names of the new columns (from "col" to "k_years_ago_col")
    new_columns = [str(k)+"_years_ago_" + column for column in columns_to_select]
    # Transform in DataFrame: DataFrame containing all the new columns. It's the DataFrame to add.
    df_to_add = pd.DataFrame(statistics_list,columns=new_columns)
    # Set as index the index of `df`
    df_to_add = df_to_add.set_index(df.index)

    df = pd.concat([df,df_to_add],axis=1) # Add the new columns

    # Rename duplicated columns
    df.columns = df.columns.map(lambda col: col.rsplit(".",1)[0] if any([(new_column in col) for new_column in new_columns])
                                                                 else col)
    df.columns = pd.io.parsers.ParserBase({'names':df.columns})._maybe_dedup_names(df.columns)

    X,y = split_X_y(df,y_col,scale_y=scale_y)

    return df, X, y


def add_current_year_statistics(df, df_current_year, days_to_select=11, current_day=False, stat="mean",
                                columns_to_select=None, replace_miss=True, y_col=None, scale_y=True):
    """
    Add, to a time series DataFrame, statistics computed on the other given time series DataFrame, with respect to the
    preceding days of the same year.

    `df_current_year` should contain days of the same year with respect to the days of `df`.(Nevertheless, both `df` and
    `df_current_year` can contain multiple years).
    Let 'm' be the number of the selected columns of `df_current_year` (by default all the columns, see the
    `columns_to_select` parameter).
    A new DataFrame is built, which is created from `df` adding 'm' new columns. (The resulting DataFrame has the same
    index of `df`). These new 'm' columns contain the values computed from the associated columns of `df_current_year`
    considering the preceding days of the same year with respect to the days in `df`.

    Going into the details, let 'day' be a row of `df`, and 'new_column' be one of the 'm' new columns created
    in the resulting DataFrame. The value put in that column for that day is computed from the associated column of
    `df_current_year` considering the preceding days of the same year, in `df_current_year`, that are centered in 'day'.
    (See the find_current_year_days function).
    Once the preceding days of the same year are selected from `df_current_year`, an unique value for the new column
    'new_column' and for the day 'day' is computed applying a certain statistical aggregation (specified by the input
    parameter `stat`) on the values of these selected days in the column of `df_current_year` associated with 'new_column'.

    `days_to_select` specifies, for each 'day' of `df`, which preceding days of the same year are selected from
    `df_current_year`. The semantics is quite similar to the parameter `n_days` of the find_current_year_days function (it
    can either be an integer or "month" or "season").

    Actually, `days_to_select` can also be more powerful than that. `days_to_select` can be a predicate (i.e. a function that
    returns a bool), which is used to select the same year days: for each 'day' of `df`, the preceding same year days of
    `df_current_year` for which the function `days_to_select` returns True are selected.
    So, `days_to_select` is a predicate that, in a flexible way, selects the same year days.
    The signature of the function must be: (day: pd.TimeStamp, df: pd.DataFrame, current_year_day: pd.TimeStamp,
    df_current_year: pd.DataFrame): bool.
    Where:
        - `day` is the current day of `df` ;
        - `df` is the given DataFrame ;
        - `day_current_year` is the preceding day of the same year contained in `df_current_year`;
        - `df_current_year` is the other given DataFrame, containing days of the same year.
    The function returns True if and only if `day_current_year` is a day that has to be selected for `day`.

    For a certain 'day' of `df` it could happen that no preceding day of the same year is selected. This means that this
    'day' has a missing value for each of the 'm' new columns (i.e. 'm' missing values).
    In this case, if `replace_miss` is True, all the missing values are filled: the missing value for the new column
    'new_column' is filled computing the mean of all the preceding days of 'day' in the associated column in
    `df_current_year`. If there isn't any preceding day in `df_current_year`, that 'day' is removed from the resulting
    DataFrame (the missing values can't be filled).
    (The removed days are surely the first days in `df`).
    Otherwise, if `replace_miss` is False, the 'm' missing values are simply kept as Nan. (No day has to be removed).

    If `current_day` is True, each 'day' of `df` is itself a potential same year day that can be selected. I.e. not only the
    preceding days are considered.
    (This is applied also in the selection of the days to be used to fill the missing values).

    So, in the end, 'm' new columns are created in the resulting DataFrame, from the selected 'm' columns of
    `df_current_year`.
    From the selected column with name "col" of `df_current_year`, the corresponding column "current_year_col" is created
    in the resulting DataFrame.

    In addition, the resulting DataFrame is automatically split into the X matrix and the y vector, which are respectively
    the matrix containing the explanatory features and the vector containing the response feature.
    (The response feature is the one which is the target of the prediction analysis tasks).

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame indexed by days (i.e. the index is a pd.DatetimeIndex).
    df_current_year: pd.DataFrame
        DataFrame indexed by days (i.e. the index is a pd.DatetimeIndex). It should contain days of the same year with
        respect to the days in `df`.
    days_to_select: int or str or callable
        Indicates, for each day of `df`, which preceding days of the same year are selected in `df_current_year`.
        It must be either an integer or "month" or "season" or a predicate (i.e. a function that returns a boolean).
        The function signature must be
                (day: pd.TimeStamp, df: pd.DataFrame, day_current_year: pd.TimeStamp, df_current_year: pd.DataFrame): bool
    current_day: bool
        Indicates if, each day of `df`, can be potentially selected for itself as a day of the same year.
    stat: str
        Indicates the statistical aggregation to perform, for each day of `df`, on the selected same year days of
        `df_current_year`.
        It can either be "mean" or "min" or "max".
    columns_to_select: list
        List of strings which indicates the columns of `df_current_year` that have to be taken into account.
        If it's None, all the columns of `df_current_year` are considered.
    replace_miss: bool
        Indicates whether to fill the missing values or keep them as Nan.
        (The missing values are generated for each day of `df` for which no same year day in `df_current_year` is selected).
    y_col: str
        Indicates which is the column of the resulting DataFrame to be used as y column.
    scale_y: bool
        Indicates whether to scale or not the values of the response feature y.

    Returns
    ----------
    pd.DataFrame
        The resulting DataFrame.
    X: np.array
        Two-dimensional numpy array which contains the explanatory features.
    y: np.array
        Mono-dimensional numpy array which contains the response feature.

    Raises
    ----------
    ValueError
        When `stat` is neither "mean" nor "min" nor "max".

    Warns
    ----------
    UserWarning
        - When, for a day of `df`, no preceding day of the same year is selected from `df_current_year`.
        - When, for a day of `df`, less preceding days of the same year are found compared to the ones expected.
          (This can happen only if `days_to_select` is either an integer or "month" or "season").

    See Also
    ----------
    find_current_year_days: returns, given a day, the selected preceding days of the same year.
    split_X_y: splits a DataFrame into X and y.

    Notes
    ----------
    If add_current_year_statistics is applied multiple times on the same `df` and `df_current_year`, columns with the same
    name are potentially created.
    For instance, if add_current_year_statistics is applied three times on the same DataFrames, from the `df_current_year`
    column "col" three different columns with the same name "current_year_col" are potentially created.
    To avoid that, add_current_year_statistics ensures that all the different columns with the same name are properly
    disambiguated, using progressive numbers. (E.g three different columns with the same name "current_year_col" became
    "current_year_col", "current_year_col.1" and "current_year_col.2").
    """

    df = df.copy()

    if not columns_to_select: # If columns_to_select is None, all the `df_k_years_ago` columns are taken into account
        columns_to_select = list(df_current_year.columns)

    # Two dimensional list (matrix-like). It contains, for each day of `df`, the values computed on `df_current_year` for
    # that day (i.e. the values to be added in the resulting DataFrame).
    # So, it's a matrix that contains as many rows as the days in `df` and as many columns as the selected columns in
    # `df_current_year`.
    statistics_list = []

    # List of days of `df` that have to be removed. (It's used only if `replace_miss` is True).
    # These are the days for which no same year day is selected and for which there aren't preceding days in
    # `df_current_year` (preceding days or also the day itself, if `current_day` is True).
    days_to_remove = []

    for day in df.index: # Iterate through all the days in `df`

         if not callable(days_to_select): # days_to_select is not a function : either an integer or "month" or "season"
             current_year_days = find_current_year_days(day, days_to_select, current_day) # Same year days (expected)
             expected_n_days = len(current_year_days) # Number of these expected same year days
             # Actual same year days in `df_current_year`
             current_year_days = current_year_days[list(current_year_days.map(lambda day_current_year: (day_current_year in
                                                                                                df_current_year.index) ))]

         else: # days_to_select is a function
             # Selected same year days (The ones filtered by the function)
             current_year_days = df_current_year.index[
                                                        list(df_current_year.index.map(lambda day_current_year: \
                                                                                          (day_current_year<day and
                                                                                          day_current_year.year==day.year and
                                                                                          days_to_select(day, df,
                                                                                                         day_current_year,
                                                                                                         df_current_year)
                                                                                          )
                                                                                        )
                                                             )
                                                       ]
             if current_day and days_to_select(day, df, day, df_current_year):
                 current_year_days.append(pd.Series([day]))

         if len(current_year_days)==0: # No same year day has been found
             warnings.warn("No current year days have been found for the day " + day.strftime('%Y-%m-%d'))
        # Less same year days have been selected than the ones expected
         elif not callable(days_to_select) and len(current_year_days)<expected_n_days:
             warnings.warn(("For the day "+day.strftime('%Y-%m-%d')+
                            " only these current year days have been found: "+
                            str(list(current_year_days.strftime('%Y-%m-%d')))))

        # No same year day has been found and `replace_miss` is True: the missing values are filled with the
        # means of the columns.
        # (If `replace_miss`, is False the missing values are kept as Nan)
         if len(current_year_days)==0 and replace_miss:
            prev_days = [ d for d in df_current_year.index if d<day ] # Days of `df_current_year` that precede 'day'
            if current_day: # If `current_day` is True, add also the day itself
                prev_days.append(day)
            # There isn't any day of `df_current_year` that precedes 'day' : this is a day to remove. (I don't know how to
            # fill the missing values).
            # (It's removed only if `replace_miss` is True)
            if len(prev_days)==0:
                days_to_remove.append(day)
            if stat=="mean":
                statistics_list.append([df_current_year[column].loc[prev_days].mean() for column in columns_to_select ])
            elif stat=="min":
                statistics_list.append([df_current_year[column].loc[prev_days].min() for column in columns_to_select ])
            elif stat=="max":
                statistics_list.append([df_current_year[column].loc[prev_days].max() for column in columns_to_select ])
            else:
                raise ValueError("stat must be a statistical measure : \"mean\" or \"min\" or \"max\" ")
         else: # There is at least one selected same year day (or `replace_miss` is False)
            if stat=="mean":
                statistics_list.append(list(df_current_year[columns_to_select].loc[current_year_days].mean()))
            elif stat=="min":
                statistics_list.append(list(df_current_year[columns_to_select].loc[current_year_days].min()))
            elif stat=="max":
                statistics_list.append(list(df_current_year[columns_to_select].loc[current_year_days].max()))
            else:
                raise ValueError("stat must be a statistical measure : \"mean\" or \"min\" or \"max\" ")

    # Names of the new columns (from "col" to "current_year_col")
    new_columns = ["current_year_" + column for column in columns_to_select]
    # Transform in DataFrame: DataFrame containing all the new columns. It's the DataFrame to add.
    df_to_add = pd.DataFrame(statistics_list,columns=new_columns)
    # Set as index the index of `df`
    df_to_add = df_to_add.set_index(df.index)

    df = pd.concat([df,df_to_add],axis=1) # Add the new columns

    df = df.drop(days_to_remove) # Remove the days that have to be removed

    # Rename duplicated columns
    df.columns = df.columns.map(lambda col: col.rsplit(".",1)[0] if any([(new_column in col) for new_column in new_columns])
                                                                 else col)
    df.columns = pd.io.parsers.ParserBase({'names':df.columns})._maybe_dedup_names(df.columns)

    X,y = split_X_y(df,y_col,scale_y=scale_y)

    return df, X, y


def add_upTo_k_years_ago_statistics(df, df_upTo_k_years_ago, k=1, current_year=True, days_to_select=11, current_day=False,
                                    stat="mean", columns_to_select=None, replace_miss=False, y_col=None, scale_y=True):
    """
    Add, to a time series DataFrame, statistics computed on the other given time series DataFrame, but with respect to the
    days of up to k years ago.

    `df_upTo_k_years_ago` should contain days of up to `k` years ago with respect to the days of `df`.(Nevertheless, `df`
    can contain multiple years).
    Let 'm' be the number of the selected columns of `df_upTo_k_years_ago` (by default all the columns, see the
    `columns_to_select` parameter).
    A new DataFrame is built, which is created from `df` adding 'm' new columns. (The resulting DataFrame has the same
    index of `df`). These new 'm' columns contain the values computed from the associated columns of `df_upTo_k_years_ago`
    considering the days of up to `k` years before the ones in `df`.

    Let 'day' be a row of `df`, and 'new_column' be one of the 'm' new columns created in the resulting DataFrame. The value
    put in that column for that day is computed from the associated column of `df_upTo_k_years_ago` considering the days of
    `df_upTo_k_years_ago` that are centered on `day` but up to `k` years ago.
    Going into the details, for each integer 'i' from 1 to `k`, the 'i' years ago days centered on 'day' and contained
    in `df_upTo_k_years_ago` are selected (see the find_k_years_ago_days function): from these selected 'i' years ago days,
    an unique value is computed applying a certain statistical aggregation (specified by the input parameter `stat`) on the
    values of these selected days in the column of `df_upTo_k_years_ago` associated to 'new_column'.
    In this way, for 'day' and 'new_column' `k` values are computed, for each integer 'i' from 1 to `k`. In the end, an
    unique value for the new column 'new_column' and for the day 'day' is computed applying the same statistical aggregation
    (i.e. `stat`) on these `k` values.
    On the whole, an aggregation with 2 levels is computed on the days of up to `k` years ago with respect to 'day'.
        - An aggregation is computed on the selected days of 'i' years ago days, for 'i' between 1 and `k`.
        - An aggregation is computed on the `k` values computed for each of the previous years, up to `k` years ago.

    Basically, this is implemented by applying `k` times the function add_k_years_ago_statistics: for each 'i' between 1 and
    `k`, add_k_years_ago_statistics is applied with his input parameter 'k' equal to 'i'.
    add_k_years_ago_statistics is applied on all the previous years, up to `k` years ago.
    So, add_upTo_k_years_ago_statistics is nothing else than an extension of the add_k_years_ago_statistics function.
    (See add_k_years_ago_statistics).
    The meaning of the input parameters `days_to_select`, `stat`, `replace_miss`, ..., are the same of the ones seen in
    add_k_years_ago_statistics.

    In particular, `days_to_select` specifies, for each 'day' of `df`, which 'i' years ago days in `df_upTo_k_years_ago` are
    selected, for 'i' from 1 to `k`. It can either be an odd integer or "month" or "season" or a predicate (i.e. a function
    that returns a bool).
    The signature of the function must be:
        (day: pd.TimeStamp, df: pd.DataFrame, day_i_years_ago: pd.TimeStamp, df_upTo_k_years_ago: pd.DataFrame): bool.

    If `current_year` is True, also the current year is taken into account, and not only the preceding years up to `k` years
    ago.
    This means that, for each 'day' of `df`, `k`+1 values are computed: from the current year; from the previous year; ...;
    from `k` years ago. These `k`+1 values are aggregated in a single value.
    The value computed from the current year is calculated using the add_current_year_statistics function (see
    add_current_year_statistics).
    The meaning of the input parameters `days_to_select`, `current_day`, `stat`, `replace_miss`, ..., are the same as the
    ones seen in add_current_year_statistics.
    (If `current_year` is True, add_current_year_statistics is applied one time and then add_k_years_ago_statistics is
    applied `k` times).

    So, in the end, 'm' new columns are created in the resulting DataFrame, from the selected 'm' columns of
    `df_upTo_k_years_ago`.
    From the selected column with name "col" of `df_upTo_k_years_ago`, the corresponding  column "upTo_k_years_ago_col" is
    created in the resulting DataFrame.

    In addition, the resulting DataFrame is automatically  split into the X matrix and the y vector, which are respectively
    the matrix containing the explanatory features and the vector containing the response feature.
    (The response feature is the one which is the target of the prediction analysis tasks).

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame indexed by days (i.e. the index is a pd.DatetimeIndex).
    df_upTo_k_years_ago: pd.DataFrame
        DataFrame indexed by days (i.e. the index is a pd.DatetimeIndex). It should contain up to `k` years ago days with
        respect to the days in `df`.
    k: int
        Indicates how many previous years have to be taken into account (i.e. all the previous years up to `k` years ago are
        taken into account).
        It must be a positive integer.
    current_year: bool
        Indicates whether to consider also the current year or not: in the former case are taken into account all the years
        from the current up to `k` years ago.
    days_to_select: int or str or callable
        Indicates, for each day of `df`, which days in `df_upTo_k_years_ago` have to be selected, for each year from the
        previous up to `k` years ago.
        It must either be an odd integer or "month" or "season" or a predicate (i.e. a function that returns a boolean).
        If `current_year` is True, this selection is also applied on the days of the same year.
    current_day: bool
        Indicates if each day of `df` can be potentially selected for itself as a day of the same year.
        This parameter is considered only if `current_year` is True.
    stat: str
        Indicates the statistical aggregation to perform. It can either be "mean" or "min" or "max".
        This aggregation is applied in two levels: both for each previous year (up to `k` years ago) and for the aggregation
        of the `k` computed values (`k`+1 if `current_year` is True).
    columns_to_select: list
        List of strings which indicates the columns of `df_upTo_k_years_ago` that have to be taken into account.
        If it is None, all the columns of `df_upTo_k_years_ago` are considered.
    replace_miss: bool
        Indicates whether to fill the missing values.
    y_col: str
        Indicates which is the column of the resulting DataFrame to be used as y column.
    scale_y: bool
        Indicates whether to scale or not the values of the response feature y.

    Returns
    ----------
    pd.DataFrame
        The resulting DataFrame.
    X: np.array
        Two-dimensional numpy array which contains the explanatory features.
    y: np.array
        Mono-dimensional numpy array which contains the response feature.

    Raises
    ----------
    ValueError
        - When `k` is not a positive integer.
        - When `stat` is neither "mean" nor "min" nor "max".

    See Also
    ----------
    find_k_years_ago_days: returns, given a day, the selected days of k years ago.
    add_k_years_ago_statistics:
        adds, to a time series DataFrame, statistics computed on the other given time series DataFrame, but with respect to
        the days of k years ago.
    find_current_year_days: returns, given a day, the selected preceding days of the same year.
    add_current_year_statistics:
        adds, to a time series DataFrame, statistics computed on the other given time series
        DataFrame, with respect to the preceding days of the same year.
    split_X_y: splits a DataFrame into X and y.

    Notes
    ----------
    - If add_upTo_k_years_ago_statistics is applied multiple times with the same `k` on the same `df` and
      `df_upTo_k_years_ago`, columns with the same name are potentially created.
      For instance, if add_upTo_k_years_ago_statistics is applied three times with the same `k` on the same DataFrames,
      from the `df_upTo_k_years_ago` column "col" three different columns with the same name "upTo_k_years_ago_col" are
      potentially created.
      To avoid that, add_upTo_k_years_ago_statistics ensures that all the different columns with the same name are properly
      disambiguated, using progressive numbers. (E.g three different columns with same name "upTo_k_years_ago_col" became
      "upTo_k_years_ago_col", "upTo_k_years_ago_col.1" and "upTo_k_years_ago_col.2").
    - The meaning of `replace_miss` is the same seen in add_k_years_ago_statistics. For each previous year up to `k` years
      ago, if no selected day is found and `replace_miss` is True, the mean of the whole `df_upTo_k_years_ago` DataFrame is
      computed: this is the value calculated for that year (value that will be aggregated with the other `k` values).
      This same concept is valid also for the current year, if `current_year` is True.
    """

    if k<0:
        raise ValueError("k must be a non-negative number")

    df = df.copy()

    if not columns_to_select: # If columns_to_select is None, all the `df_upTo_k_years_ago` columns are taken into account
        columns_to_select = list(df_upTo_k_years_ago.columns)

    cols_to_skip = list(df.columns) # Take the `df` columns, which are the columns to skip

    # List that will contain `k`+1 DataFrames : they are the DataFrames which have the same days of `df` (i.e. same index)
    # and as columns the selected columns of `df_upTo_k_years_ago`, with respect to the different `k`+1 years.
    # I.e. the first DataFrame is related to the current year, the second to the previous year, the third to two years ago,
    # ... .
    # The current year is taken into account only if `current_year` is True.
    dfs_to_add = []

    if current_year: # current_year is True : the current year is taken into account
        dfs_to_add.append(add_current_year_statistics(df, df_current_year=df_upTo_k_years_ago, days_to_select=days_to_select,
                                                      current_day=current_day, stat=stat,
                                                      columns_to_select=columns_to_select, replace_miss=replace_miss,
                                                      y_col=y_col, scale_y=scale_y)[0])

    # Now add the other `k` DataFrames
    for i in range(1,k+1): # Iterate through all the past years
        # Add the DataFrame which have the same index of `df` and as columns the selected columns of `df_upTo_k_years_ago`,
        # with respect to 'i' years ago.
        dfs_to_add.append(add_k_years_ago_statistics(df, df_upTo_k_years_ago, k=i, days_to_select=days_to_select, stat=stat,
                        columns_to_select=columns_to_select, replace_miss=replace_miss, y_col=y_col, scale_y=scale_y)[0])

    # Concatenate the `k`+1 DataFrames in a single DataFrame, which have the same days of `df` (i.e. same index). It has all
    # the selected columns of `df_upTo_k_years_ago`, repeated `k`+1 times (`k` if `current_year` is False).
    # Each selected column of `df_upTo_k_years_ago` is repeated `k`+1 times, one for each year.
    # The column "col" of `df_upTo_k_years_ago` is contained `k`+1 times, with the following names: "current_year_col",
    # "1_years_ago_col", "2_years_ago_col", .... , "k_years_ago_col".
    df_supp = pd.concat(dfs_to_add,axis=1)

    # Iterate through all the selected columns of `df_upTo_k_years_ago`. For each column "col", create the corresponding
    # "upTo_k_years_ago_col" column in the resulting DataFrame.
    # In order to create "upTo_k_years_ago_col", all the `k`+1 columns in 'df_supp' related to "col" are taken (
    # "current_year_col", "1_years_ago_col", "2_years_ago_col", ... , "k_years_ago_col") and a certain aggregation is
    # computed  ("mean","min","max").
    new_columns = []
    for col in columns_to_select:
        associated_columns = [str(i)+"_years_ago_"+col for i in range(1,k+1)]
        if current_year: # The column "current_year_col" has to be considered only if `current_year` is True
            associated_columns.append("current_year_"+col)
        new_column = "upTo_"+str(k)+"_years_ago_"+col
        new_columns.append(new_column)
        # Compute the single new column, and add it to the resulting DataFrame
        if stat=="mean":
            df_to_add = pd.DataFrame({new_column:df_supp[associated_columns].mean(axis=1)})
            df = pd.concat([df,df_to_add],axis=1)
        elif stat=="min":
            df_to_add = pd.DataFrame({new_column:df_supp[associated_columns].min(axis=1)})
            df = pd.concat([df,df_to_add],axis=1)
        elif stat=="max":
            df_to_add = pd.DataFrame({new_column:df_supp[associated_columns].max(axis=1)})
            df = pd.concat([df,df_to_add],axis=1)
        else:
            raise ValueError("stat must be a statistical measure : \"mean\" or \"min\" or \"max\",")

    # Rename the duplicated columns
    df.columns = df.columns.map(lambda col: col.rsplit(".",1)[0] if any([(new_column in col) for new_column in new_columns])
                                                                 else col)
    df.columns = pd.io.parsers.ParserBase({'names':df.columns})._maybe_dedup_names(df.columns)

    X,y = split_X_y(df,y_col,scale_y=scale_y)

    return df, X, y
