import re

def natural_key(string):
    """
    Splits a string into a list of strings and numbers for natural sorting.

    Parameters:
        string (str): The input string that may contain digits.

    Returns:
        list: A list containing substrings and integer conversions of digits,
              allowing for natural (human-friendly) sorting.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split("([0-9]+)", string)]
