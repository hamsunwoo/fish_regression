import os

def get_model_path():
    f= __file__

    dir_name = os.path.dirname(f)
    model_path = os.path.join(dir_name, "regression.pkl")
    
    return model_path
