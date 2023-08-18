""" Creates init files for every 
directory and puts directory name
into docstring.

Run from command line:
python3 scripts/utilities/python_create_inits.py
"""

import os

def create_init_file(path):
    init_path = os.path.join(path, '__init__.py')
    if not os.path.exists(init_path):
        with open(init_path, 'w') as init_file:
            init_file.write(f'"""{path}"""\n')

def main():
    repo_path = os.getcwd()  # Gets the current working directory
    for root, dirs, files in os.walk(repo_path):
        create_init_file(root)

if __name__ == "__main__":
    main()
