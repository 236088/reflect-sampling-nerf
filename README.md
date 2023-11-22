## File Structure
We recommend the following file structure:

```
├── reflect_sampling_nerf
│   ├── __init__.py
│   ├── reflect_sampling_nerf_config.py
│   ├── reflect_sampling_nerf_pipeline.py
│   ├── reflect_sampling_nerf_model.py
│   ├── reflect_sampling_nerf_field.py
│   ├── reflect_sampling_nerf_datamanger.py
│   ├── reflect_sampling_nerf_dataparser.py
│   ├── ...
├── pyproject.toml
```

## Registering with Nerfstudio
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html). Clone or fork this repository and run the commands:

```
conda activate nerfstudio
cd reflect-sampling-nerf/
pip install -e .
ns-install-cli
```

## Running the new method
This repository creates a new Nerfstudio method named "reflect-sampling-nerf". To train with it, run the command:
```
ns-train reflect-sampling-nerf --data [PATH]
```