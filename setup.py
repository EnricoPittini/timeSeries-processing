from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
  name="timeSeries-processing",
  version="0.0.2",
  py_modules =["timeSeries_processing"],
  description="Library which processes time series datasets",
  long_description=long_description,
  long_description_content_type='text/markdown',
  url="https://github.com/EnricoPittini/timeSeries-processing",
  author="Enrico Pittini",
  author_email="pittinienrico@hotmail.it",
  license="MIT",
  install_requires=['matplotlib',
                    'pandas',
                    'sklearn' ],
  classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
)
