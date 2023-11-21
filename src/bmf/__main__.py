import logging

from . import model
from . import baselines
from . import timer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# get model
t = timer.Timer()
data = model.read_data_ml1m()
train_data = model.ds_to_df(data["train"])
eval_data = model.ds_to_df(data["validation"])
logger.info("Loaded the data in %.3fs", t.lap())
logger.info("  Train data: %s", str(model.df_analyze(train_data)))
logger.info("  Eval data: %s", str(model.df_analyze(eval_data)))

# run evaluation
for baseline in baselines.baseline_methods.values():
    predictor = baseline(train_data)
    logger.info("Trained the %s in %.3fs", str(predictor), t.lap())
    rmse = predictor.rmse(eval_data)
    logger.info("Evaluation of %s in %.3fs: RMSE=%f", str(predictor), t.lap(), rmse)
