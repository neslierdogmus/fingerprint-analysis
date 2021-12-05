import hashlib
import papermill as pm
import nbformat
from nbconvert import HTMLExporter

parameters = dict(dataset_dir='datasets/Finger/FOESamples',
                  output_dir='results',
                  split_id=2,
                  num_folds=5,
                  patch_size=64,
                  batch_size=64,
                  hflip=True,
                  rotate=True,
                  use_cpu=False,
                  seed=0,

                  num_classes=36,
                  num_epochs=100,
                  learning_rate=0.0001,
                  approach='cnn',

                  encoded_space_dim=512,
                  train_with_bad=True,
                  ae_num_epochs=300,
                  ae_learning_rate=0.1)

file_id = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()
parameters['file_id'] = file_id

pm.execute_notebook(
   'experiment.ipynb',
   'experiment.ipynb',
   cwd='.',
   kernel_name='python3',
   parameters=parameters)

# read source notebook
with open('experiment.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

# export to html
html_exporter = HTMLExporter()
html_exporter.exclude_input = True
html_data, resources = html_exporter.from_notebook_node(nb)

# write to output file
with open("notebook.html", "w") as f:
    f.write(html_data)
