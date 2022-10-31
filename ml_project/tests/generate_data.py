"""
Source:
https://github.com/DataResponsibly/DataSynthesizer/blob/master/notebooks/DataSynthesizer__correlated_attribute_mode.ipynb
"""

from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # input dataset
    input_data = 'data/raw/heart_cleveland_upload.csv'
    # location of two output files
    mode = 'correlated_attribute_mode'
    description_file = 'tests/data/description.json'
    synthetic_data = 'tests/data/synthetic_data.csv'

    threshold = 5

    # specify categorical attributes
    categorical_attributes = {
        'sex': True,
        'cp': True,
        'fbs': True,
        'restecg': True,
        'exang': True,
        'ca': True,
        'thal': True
    }

    epsilon = 0
    degree_of_bayesian_network = 2
    num_tuples_to_generate = 150
    describer = DataDescriber(category_threshold=threshold)
    describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data,
                                                            epsilon=epsilon,
                                                            k=degree_of_bayesian_network,
                                                            attribute_to_is_categorical=categorical_attributes)
    describer.save_dataset_description_to_file(description_file)
    display_bayesian_network(describer.bayesian_network)
    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
    generator.save_synthetic_data(synthetic_data)

    # Read both datasets using Pandas.
    input_df = pd.read_csv(input_data, skipinitialspace=True)
    synthetic_df = pd.read_csv(synthetic_data)
    # Read attribute description from the dataset description file.
    attribute_description = read_json_file(description_file)['attribute_description']

    inspector = ModelInspector(input_df, synthetic_df, attribute_description)

    for attribute in synthetic_df.columns:
        inspector.compare_histograms(attribute)
        plt.suptitle(f'{attribute.upper()}')
        plt.savefig(f'tests/data/statistics/{attribute}_histogram.png')

    inspector.mutual_information_heatmap()
    plt.savefig('tests/data/statistics/heatmap.png')
