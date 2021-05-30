# Tutorials

## D-Wave Setup
1. Install pyQBTNs:
    - Option 1: Install using pip (also download the [example D-Wave code](../tests/TestMatrixFactorizationQuantum.py) from repository)
    ```shell
    pip install git+https://gitlab.lanl.gov/epelofske/qbtns/-/tree/master
    ```
    - Option 2: Install from source
    ```shell
    git clone https://gitlab.lanl.gov/epelofske/qbtns/-/tree/master
    cd qbtns
    conda create --name pyQBTNs python=3.7.3
    source activate pyQBTNs
    python setup.py install
    ```

2. Sign up with [D-Wave Leap](https://cloud.dwavesys.com/leap/signup/).
    - Make sure that you have at least 1 minute of QPU time on your free acccount.
3. Set up [D-Wave config file](https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html):
    - Configuration file can be created using the command line tool. It will prompt several questions:
    ```shell
    dwave config create
    ```
    
    - It will ask for configuration path. Provide the path or leave empty and press enter: 
    ```shell
    Configuration file path:
    ```
    
    - Type ```new``` if it asks for *Profile* or choose from the provided list: 
    ```shell
    Profile (create new or choose from: prod): new
    ```
    
    - Next, it will ask for API entpoint URL. You can get the URL from *D-Wave* dashboard:
    ```shell
    API endpoint URL [skip]: https://cloud.dwavesys.com/sapi/
    ```
    
    - Next, it will prompt for authentication token. Your token can be found in *D-Wave* dashboard under *API Token*:
    ```shell
    Authentication token [skip]:TOKEN
    ```
    
    - After the token is provided, you will be prompted to enter a client class name. We will use *qpu*:
    ```
    Default client class [skip]: qpu
    ```
    
    - Now we need to enter the solver name. You can choose a solver from your *D-Wave* solver dashboard under *Available Solvers*. Since we are using the *QPU* class, example solver names could be *Advantage_system1.1* or *DW_2000Q_6*. Note that these names could change; therefore, see the *D-Wave* dashboard to get the solver name:
    ```shell
    Default client class [skip]: DW_2000Q_6
    ```
4. Run an example (download the exampe from [here](../tests/TestMatrixFactorizationQuantum.py)):
```shell
python -m unittest TestMatrixFactorizationQuantum.py
```

**Note** that if the installation is done from the source in **step 1**, example code in **step 4** can be run from the *tests* directory after ```cd tests```.
