# Logistic Regression
2031561 Chaozheng Guo  
Python implementation of logistic regression.  

### Packages reqired
```numpy``` ```argparse```

### Usage
```python logistic_regression.py```  
By default, this will load breast cancer dataset and perform logistic regression. Model prediction score will be displayed.


Some options can be specified for the program.  
```-dataset``` str, Dataset to be used, breast_cancer or abalone.  
```-e``` flost, Threshold for coefficient update.  
```-max_it``` int, Maximum iteration time.

Example:
```python logistic_regression.py -dataset abalone -e 1e-2 -max_it 5```  





