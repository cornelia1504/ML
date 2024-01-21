''' Main script for cancer classification prediction model '''

#import files
import lib
#############

if __name__ == '__main__':
    rna_file = 'data/TCGA-PANCAN-HiSeq-801x20531/TCGA-PANCAN-HiSeq-801x20531/data.csv'
    label_file = 'data/TCGA-PANCAN-HiSeq-801x20531/TCGA-PANCAN-HiSeq-801x20531/labels.csv'

    print('Random forest ... process ...')
    lib.random_forest_process(rna_file, label_file)
    print('Random forest ... done ...')

    print('logistic_regression ... process ...')
    lib.logistic_regression_process(rna_file, label_file)
    print('logistic_regression ... done ...')

    print('neural_network ... process ...')
    #lib.neural_network_process(rna_file, label_file)
    print('neural_network ... done ...')

    print('Thank you for using the programme. All plots and files are available in the output directory ')

