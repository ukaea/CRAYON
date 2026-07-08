# Installation #

## Local installation from sources ##

We recommend using a venv. This can be created and activated using

```
python3 -m venv ./venv.
. venv/bin/activate
```

Clone the Crayon respository, cd into the root directory and run *pip install*

```
pip install .
```

Test the installation by trying

```
python -c "import crayon; print(crayon.__version__)"
```
