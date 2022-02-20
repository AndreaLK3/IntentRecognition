import Filepaths
import Utils
import os
import Model.Training as T
import pandas as pd

if __name__=="__main__":
    Utils.init_logging("EvaluationOnTestSet_" + Filepaths.LATEST_BEST_MODEL_FNAME[:-3] + ".log")

    model = Utils.load_model_from_file(Filepaths.LATEST_BEST_MODEL_FNAME)

    _, _, test_split = Utils.load_dataset()
    test_df = pd.DataFrame(test_split, columns=[Utils.UTTERANCE, Utils.INTENT])
    T.evaluation(test_df, model)
