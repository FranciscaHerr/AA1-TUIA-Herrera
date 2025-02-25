import joblib
import pandas as pd
import logging
from sys import stdout
from clases import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logformatter = logging.Formatter("%(asctime)s: %(levelname)s: %(filename)s: %(message)s")
consoleHandler = logging.StreamHandler(stdout)
consoleHandler.setFormatter(logformatter)
logger.addHandler(consoleHandler)

pipeline = joblib.load('pipeline.pkl')

logger.info('loaded pipeline')

input = pd.read_csv('/temp/df_test.csv')
input.drop(['RainTomorrow', 'RainfallTomorrow'], axis=1, inplace=True)

logger.info('loaded input')

output = pipeline.predict(input)

logger.info('made predictions')

pd.DataFrame(output, columns=['RainfallTomorrow_predicted']).to_csv('/temp/output.csv', index=False)

logger.info('saved output')
