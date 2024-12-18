# Advanced Data Processing
Our library also supports a powerful data processing backed which can be used by the users to perform custom data preprocessing including
1. Providing multiple datasets
1. Creating custom data processing pipeline for the datasets.
1. Combining multiple datasets into one with even differnt formats.
1. Mixing datasets as requried and sampling if needed each with different weights.

These things are supported via what we call a [`data_config`](#data-config) which can be passed an an argument to sft trainer. We explain data config in detail next,

## Data Config

Data config is a configuration file which users can provide to sft trainer.py

What is data config schema 

How can use write data configs

What are data handlers

Preexisting data handlers

Extra data handlers

How can use pass the datasets 

What kind of datasets can be passed

How can user perform sampling
 - What does sampling means?
 - How will it affect the datasets

How can user create a data config for the existing use cases.

Corner cases which needs attention.