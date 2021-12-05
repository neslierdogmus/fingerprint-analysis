import hashlib
import papermill as pm
import nbformat
from nbconvert import HTMLExporter

parameters = dict(dataset_dir='../datasets/Finger/FOESamples',
                  output_dir='../results',
                  use_cpu=False,
                  seed=0,
                  split_id=2,
                  num_folds=5,
                  approach='cnn',
                  batch_size=64,

                  hflip=True,
                  rotate=True,
                  num_epochs=100,
                  learning_rate=0.001)

for num_classes in [9, 18, 36, 60]:
    for patch_size in [16, 32, 64, 96]:
        new_params = dict(num_classes=num_classes, patch_size=patch_size)
        parameters.update(new_params)

        file_id = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()
        parameters['file_id'] = file_id

        pm.execute_notebook('experiment.ipynb', 'experiment_o.ipynb', cwd='.',
                            kernel_name='python3', parameters=parameters)

        # read source notebook
        with open('experiment_o.ipynb') as f:
            nb = nbformat.read(f, as_version=4)

        # export to html
        html_exporter = HTMLExporter()
        html_exporter.exclude_input = True
        html_data, resources = html_exporter.from_notebook_node(nb)

        # write to output file
        with open("{}.html".format(file_id), "w") as f:
            f.write(html_data)
