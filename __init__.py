import os
import folder_paths
from .node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

folder_paths.add_model_folder_path("FBCNN", os.path.join(folder_paths.models_dir, "FBCNN"))

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
