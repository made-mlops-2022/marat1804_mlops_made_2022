import os

import click
import pandas as pd
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.lib.utils import display_bayesian_network


INPUT_DATA = 'input_data.csv'


@click.command('generate')
@click.option('--output', type=click.Path(),
              help='Path to output files')
def generate_data(output):
    os.makedirs(output, exist_ok=True)
    input_data = os.path.join(output, INPUT_DATA)
    generate_new_data(input_data)

    data = pd.read_csv(input_data)
    target_column = data.columns[-1]
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    X.to_csv(os.path.join(output, 'data.csv'), index=False)
    y.to_csv(os.path.join(output, 'target.csv'), index=False)
    os.remove(input_data)


def generate_new_data(output_data):
    input_data = INPUT_DATA
    description_file = 'description.json'
    threshold = 5
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
    num_tuples_to_generate = 250
    describer = DataDescriber(category_threshold=threshold)
    describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data,
                                                            epsilon=epsilon,
                                                            k=degree_of_bayesian_network,
                                                            attribute_to_is_categorical=categorical_attributes)
    describer.save_dataset_description_to_file(description_file)
    display_bayesian_network(describer.bayesian_network)
    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
    generator.save_synthetic_data(output_data)
    os.remove(description_file)


if __name__ == '__main__':
    generate_data()
