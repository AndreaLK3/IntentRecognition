import Model.Training as T
import argparse
import pandas as pd
import Utils

def parse_inference_arguments():
    parser = argparse.ArgumentParser(description='Train several models with different hyperparameters. '
                                                 'Optionally, evaluate them on the test set ')

    parser.add_argument('--with_evaluation', type=bool, default=False,
                        help='Once trained a model, perform inference on the test set')

    args = parser.parse_args()
    return args

# (batch_size, learning_rate)
hyperparams_lts = [(16, 5e-4)]# [(2,5e-5), (4,1e-4), (4,5e-4), (8,1e-4), (8,5e-4), (16,5e-4)]


if __name__ == "__main__":

    args = parse_inference_arguments()
    for (bsz, lr) in hyperparams_lts:
        model = T.run_train(learning_rate=lr, batch_size=bsz)

        if args.with_evaluation == True:
            _, _, test_split = Utils.load_dataset()
            test_df = pd.DataFrame(test_split, columns=[Utils.UTTERANCE, Utils.INTENT])
            T.evaluation(test_df, model)
