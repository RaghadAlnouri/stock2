from google.cloud import automl_v1beta1 as automl
import numpy as np
import requests
from google.api import httpbody_pb2
from google.cloud import aiplatform as aip
from google.cloud import aiplatform_v1 as gapic
import json

from google.api import httpbody_pb2
from google.cloud import aiplatform_v1

from src.IO.get_data import create_data_fetcher
from src.IO.storage_tools import get_model_from_bucket, upload_file_to_bucket, create_bucket
from src.algo.create_pipeline import create_predictor
from src.business_logic.constants import NUM_LAGS, ROOT_BUCKET


def get_prediction(ticker):
    DATA = {
    "signature_name": "predict",
    "instances": [
        {"formatted_date": "2022-11-16",
                "high": "116.80999755859376",
                "low":"113.2300033569336",
                "open":"115.0",
                "close":"115.01000213623048",
                "volume":"2081600.0",
                "ticker":ticker,
        }
    ],
    }
    endpoint = aip.Endpoint('projects/nlp-course-362518/locations/us-central1/endpoints/799322963160596480')

    http_body = httpbody_pb2.HttpBody(
        data=json.dumps(DATA).encode("utf-8"),
        content_type="application/json",
    )
    req = aiplatform_v1.RawPredictRequest(
        http_body=http_body, endpoint=endpoint.resource_name
    )
    API_ENDPOINT = "{}-aiplatform.googleapis.com".format(REGION)
    client_options = {"api_endpoint": API_ENDPOINT}

    pred_client = aip.gapic.PredictionServiceClient(client_options=client_options)

    response = pred_client.raw_predict(req)
    return response
    