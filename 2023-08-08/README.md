# Instructions for today's session
You will get a link to the slides during the call.

## Prerequisites
- [ ] Google account
- [ ] Install docker (if you want to run a few things locally) / manage python environment
- [ ] DAGWorks account if you want to access the UI (& API key); we can help here.
- [ ] openAI API key (if you want to run the PDF summarizier example); we can help here.

## Setting up Google Colab
1. Mount google drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
2. Change directory to google drive.
```bash
%cd /content/drive/MyDrive
```
3. Make a directory "hamilton-tutorials"
```bash
!mkdir hamilton-tutorials
```
4. Change directory to it.
```bash
%cd hamilton-tutorials
```
5. Clone this repository to your google drive
```bash
!git clone https://github.com/DAGWorks-Inc/hamilton-tutorials/
````
6. Move your current directory to the hello_world example
```bash
%cd hamilton-tutorials/hello_world
````
7. Install requirements.
```bash
%pip install -r requirements.txt
```
To check your current working directory you can type `!pwd` in a cell and run it.

## Iterating on Hamilton code
You can either create modules manually, or use the `%%writefile my_module.py` magic command to write a file to disk.
```python
%%writefile my_module.py

import foo

def my_func():
    return foo.bar()
```

You will then want to reload the module with `importlib.reload(my_module)` if you make changes, or use the
%aimport magic command to automatically reload modules. E.g.:
```python
import importlib
import my_module
importlib.reload(my_module)
```
or
```python
# import the jupyter extension
%load_ext autoreload
# set it to only reload the specified modules
%autoreload 1
# specify the Python modules to reload
%aimport functions, functions2
# for more info: https://ipython.org/ipython-doc/3/config/extensions/autoreload.html
```

## Hello World
See the hello_world directory for a simple example of how to use Hamilton.

## Connecting with DAGWorks
1. Create an account on DAGWorks.
2. Create an API key. Take note of it.
3. Create a new project and find the "using Hamilton" part. Cut and paste it.
```python
dwdr = dw_driver.Driver(
    config,
    your_modules,  # python module containing function logic
    adapter=base.SimplePythonGraphAdapter(base.DictResult()), # optional
    project_id=YOUR_PROJECT_ID
    api_key=DAGWORKS_API_KEY,
    username="your@email",
    dag_name="NAME_FOR_CURATION",
    tags={"env": "local", "origin": "notebook"}
)
```
4. Hit execute and things should show up/work in the DAGWorks Platform UI.

## More advanced examples
Ordered by how much code you need to write.

### Hamilton for RAG (LLM) workflow
PDF summarizer example: want to play with building LLM dataflows? See the pdf_summarizer directory.
Extensions:
* Reimplement in Hamilton some Langchain chains.
* Spin up the docker app locally and play with it & make modifications; connect it with the DAGWorks UI.

### Hamilton for end-to-end ML
The credit card approval example should get you started on how one might use Hamilton for end-to-end ML.

### Hamilton for feature engineering
We have a pretty open slate here. We have some more datasets that you could play with:

1. Absenteeism at work.
2. Inventory forecasting.

For more inspiration, see [Hamilton + Feast](https://github.com/DAGWorks-Inc/hamilton/tree/main/examples/feast),
and [general feature engineering with Hamilton](https://github.com/DAGWorks-Inc/hamilton/tree/main/examples/feature_engineering_multiple_contexts).


