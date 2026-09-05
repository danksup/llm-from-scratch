from pathlib import Path
from typing import Literal
import warnings
import datetime


class Logger:
    def __init__(self, filename, folder_path:str|Path|None=None) -> None:
        base_dir = ""
        if folder_path is not None:
            folder_path = Path(folder_path)
            if folder_path.is_dir():
                base_dir = folder_path
            else:
                base_dir = Path.cwd()
        else:
            base_dir = Path.cwd()
    
        self.filepath = base_dir / f"log_{filename}.log"

    def warn(self, msg,log_msg=None, category=None):
        if category is None or not issubclass(category, Warning):
            category = Warning
        self.log(msg, 'warn', log_msg=log_msg, category=category)
        warnings.warn(msg, category=category)

    def error(self, msg,log_msg=None, category=None):
        if category is None:
            category = RuntimeError(msg)

        cat_class = category
        if isinstance(category, BaseException):
            cat_class = category.__class__
            self.log(msg, 'raise', log_msg=log_msg, category=cat_class)
            raise category
        else:
            if isinstance(category, type) and issubclass(category, BaseException):
                self.log(msg, 'raise', log_msg=log_msg, category=category)
                raise category(msg)
            else:
                self.log(msg, 'raise', log_msg=log_msg, category=RuntimeError)
                raise RuntimeError(msg)
        
    def info(self, msg,log_msg=None, print_msg:bool=False):
        if print_msg:
            print(msg)
        self.log(msg, 'info', log_msg=log_msg)

    def log(self, msg, action:Literal['info', 'warn', 'raise']="info", category=None, log_msg=None):
        match action:
            case "warn":
                action_type = "WARN"
                action_msg = f"{action_type}: {category.__qualname__}"
            case 'raise':
                action_type = "ERROR"
                action_msg = f"{action_type}: {category.__qualname__}"
            case 'info':
                action_type = action_msg = 'INFO'

        now = datetime.datetime.now()
        time_string = now.strftime("%Y-%m-%d %H:%M:%S")

        total_msg = msg + "\n" + log_msg if log_msg is not None else msg
        log_msg = f"[{time_string}] [{action_msg}]: {total_msg}\n"
        with open(self.filepath, "a", encoding='utf-8') as f:
            f.write(log_msg)