from setuptools import setup

setup(
  name="timeSeries-processing",
  version="0.0.1",
  py_modules =["timeSeries_processing"],
  description="Library which processes time series datasets",
  url="https://github.com/EnricoPittini/timeSeries-processing",
  author="Enrico Pittini",
  author_email="pittinienrico@hotmail.it",
  license="MIT",
  install_requires=['numpy',
                    'matplotlib',
                    'pandas',
                    'ml-model-selection' ],
)
