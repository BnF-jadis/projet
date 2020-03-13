import json, glob, os, gc, argparse, cv2
import tensorflow as tf
import pandas as pd
import numpy as np

from tqdm import trange
from sacred import Experiment

from utils.utils import *
from utils.segment import makeImagePatches, removeSmallComponents

from utils.external.dhsegment import estimator_fn, utils
from utils.external.dhsegment.io import input


print('\n','Description :')
print("Entraînement d'un nouveau réseau neuronal", '\n')

print('\n','Options :')
print("--continue : reprendre l'entraînement là où il avait été laissé")

parser = argparse.ArgumentParser()
parser.add_argument('--continue', action="store_true", dest="continue_", default=False)
options = parser.parse_args()

project_name = getProjectName()

with open(os.path.join('settings', project_name, 'settings.json')) as settings:
    settings = json.load(settings)


labels_paths = glob.glob(os.path.join('model', 'train', project_name, 'images', '*.*'))
images_paths = glob.glob(os.path.join('model', 'train', project_name, 'labels', '*.*'))
class_file_path = os.path.join('model', 'train', project_name, 'text.txt')
classes = settings["segment"]["classes"]

# Clean labels annotations
import cv2, os, glob, tqdm

project_name = getProjectName()

for folder in [os.path.join('model', 'train', project_name, 'labels'),
               os.path.join('model', 'train', project_name, 'eval', 'labels')]:
    
    paths = glob.glob(os.path.join(folder, '*.*'))

    for path in tqdm.tqdm(paths, desc='Nettoyage des annotations'):
        
        img = cv2.imread(path)
        orig = img.copy()
        img[img < 255/2] = 0
        img[img >= 255/2] = 255
        img = removeSmallComponents(img, component_min_area = 20)
        img = grayToColor(orig, img)
        
        cv2.imwrite(path, img)


with open(class_file_path, 'w') as outfile:
    classes_text = ''
    for class_ in classes:
        for channel in class_:
            classes_text += str(channel) + ' '
        classes_text = classes_text[:-1] + '\n'
    classes_text = classes_text[:-1]
    outfile.write(classes_text)


params = settings['cnn']['training_params']

if len(params['weights_labels']) != len(classes):
    params['weights_labels'] = np.ones(len(classes)).astype('int').tolist()

params['exponential_learning'] = True
params['make_patches'] = False
params['patch_shape'] = (1000, 1000)
params['input_resized_size'] = int(72e4)
params['weights_evaluation_miou'] = params['weights_labels']
params['training_margin'] = 16
params['local_entropy_ratio'] = 0.
params['local_entropy_sigma'] = 3
params['focal_loss_gamma']= 0.

ex = Experiment('dhSegment_experiment')

@ex.config
def default_config():
    train_data = os.path.join('model', 'train', project_name)  # Directory with training data
    eval_data = os.path.join('model', 'train', project_name, 'eval')  # Directory with validation data
    model_output_dir = os.path.join('model', project_name + '_model')  # Directory to output tf model
    restore_model = options.continue_
    classes_file = class_file_path  # txt file with classes values
    gpu = settings['cnn']['gpu']  # GPU to be used for training
    prediction_type = utils.PredictionType.CLASSIFICATION
    pretrained_model_name = 'resnet50'
    model_params = utils.ModelParams(pretrained_model_name=pretrained_model_name).to_dict()  # Model parameters
    training_params = params  # Training parameters
    model_params['n_classes'] = len(classes)

@ex.automain
def run(train_data, eval_data, model_output_dir, gpu, training_params, _config):
    # Create output directory
    if not os.path.isdir(model_output_dir):
        os.makedirs(model_output_dir)
    else:
        assert _config.get('restore_model'), \
            '{0} already exists, you cannot use it as output directory. ' \
            'Set "restore_model=True" to continue training, or delete dir "rm -r {0}"'.format(model_output_dir)
    
    # Save config
    with open(os.path.join(model_output_dir, 'config.json'), 'w') as f:
        json.dump(_config, f, indent=4, sort_keys=True)

    # Create export directory for saved models
    saved_model_dir = os.path.join(model_output_dir, 'export')
    if not os.path.isdir(saved_model_dir):
        os.makedirs(saved_model_dir)

    training_params = utils.TrainingParams.from_dict(training_params)

    session_config = tf.ConfigProto()
    session_config.gpu_options.visible_device_list = str(gpu)
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    estimator_config = tf.estimator.RunConfig().replace(session_config=session_config,
                                                        save_summary_steps=10,
                                                        keep_checkpoint_max=1)
    estimator = tf.estimator.Estimator(estimator_fn.model_fn, model_dir=model_output_dir,
                                       params=_config, config=estimator_config)

    def get_dirs_or_files(input_data):
        if os.path.isdir(input_data):
            image_input, labels_input = os.path.join(input_data, 'images'), os.path.join(input_data, 'labels')
            # Check if training dir exists
            assert os.path.isdir(image_input), "{} is not a directory".format(image_input)
            assert os.path.isdir(labels_input), "{} is not a directory".format(labels_input)

        return image_input, labels_input

    train_input, train_labels_input = get_dirs_or_files(train_data)
    if eval_data is not None:
        eval_input, eval_labels_input = get_dirs_or_files(eval_data)

    # Configure exporter
    serving_input_fn = input.serving_input_filename(training_params.input_resized_size)
    if eval_data is not None:
        exporter = tf.estimator.BestExporter(serving_input_receiver_fn=serving_input_fn, exports_to_keep=1)
    else:
        exporter = tf.estimator.LatestExporter(name='SimpleExporter', serving_input_receiver_fn=serving_input_fn,
                                               exports_to_keep=5)

    for i in trange(0, training_params.n_epochs, training_params.evaluate_every_epoch, desc='Evaluated epochs'):
        estimator.train(input.input_fn(train_input,
                                       input_label_dir=train_labels_input,
                                       num_epochs=training_params.evaluate_every_epoch,
                                       batch_size=training_params.batch_size,
                                       data_augmentation=training_params.data_augmentation,
                                       make_patches=training_params.make_patches,
                                       image_summaries=True,
                                       params=_config,
                                       num_threads=32))

        if eval_data is not None:
            eval_result = estimator.evaluate(input.input_fn(eval_input,
                                                            input_label_dir=eval_labels_input,
                                                            batch_size=1,
                                                            data_augmentation=False,
                                                            make_patches=False,
                                                            image_summaries=False,
                                                            params=_config,
                                                            num_threads=32))

        else:
            eval_result = None

        exporter.export(estimator, saved_model_dir, checkpoint_path=None, eval_result=eval_result,
                        is_the_final_export=False)

    # If export directory is empty, export a model anyway
    if not os.listdir(saved_model_dir):
        final_exporter = tf.estimator.FinalExporter(name='FinalExporter', serving_input_receiver_fn=serving_input_fn)
        final_exporter.export(estimator, saved_model_dir, checkpoint_path=None, eval_result=eval_result,
                              is_the_final_export=True)
    


print('__________________________________________________________________________ \n')


