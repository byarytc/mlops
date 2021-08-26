from mlflow.tracking import MlflowClient
from fastai.vision import *
from fastai.metrics import error_rate
import logging
import mlflow
import json


def print_auto_logged_info(data):
    tags = {k: v for k, v in data.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(data.info.term, "model")]
    print("term: {}".format(data.info.term))
    print("artifacts: {}".format(artifacts))
    print("parameters: {}".format(data.data.parameters))
    print("metrics: {}".format(data.data.metrics))
    print("tags: {}".format(tags))



    remote_server_uri = "http://mlops_mlflow-server_1:5000"
    mlflow.set_tracking_uri(remote_server_uri)

    parameters = {
    'size': 198,
    'epochs': 4,
    'bs': 8
    }

    path_img = '/training/data'
    fnames = get_image_files(path_img)
    np.random.seed(2)
    pat = r'/([^/]+)_\d+.jpg$'
    mlflow.set_experiment("catsdogsrabbits-experiment")

  

    mlflow.fastai.autolog()

    data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=parameters['size'], bs=parameters['bs']).normalize(imagenet_stats)
    learn = cnn_learner(data, models.resnet34, metrics=error_rate)

    learn.load('/training/models/best_resnet34',strict=False,remove_module=True)

    with mlflow.start_run() as run:
        learn.fit_one_cycle(parameters['epochs'])

    logging.info("end fit")

    learn.save('/training/models/stage-1')

    learn.export('/training/models/stage-1.pkl')

    print_auto_logged_info(mlflow.get_run(term=run.info.term))

    return {'success': 'new result is ready'}, 200
