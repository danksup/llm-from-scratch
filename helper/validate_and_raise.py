from typing import Any

def get_i_type(l:list) -> str:
    msg = ""
    for i in l:
        msg += f"{i}({type(i).__name__}), "
    return msg
def validate_choice(input_choice:Any, input_name:Any, choices:list|dict|tuple|set, prefix=None):
    if input_choice not in choices:        
        choices_str = " Possible input: "+ get_i_type(list(choices)) if choices else ""
        insert_prefix = prefix + " " if prefix is not None else ""
        raise ValueError(f"{insert_prefix}Invalid input for {input_name}: \"{input_choice}\" of type {type(input_choice).__name__}.{choices_str}")

def validate_match(arg1:Any, arg2:Any, msg):
    if arg1 != arg2:
        raise ValueError(f"mismatch {msg}")