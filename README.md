***Note to Lab05***
If we want to overtrain our model on purpose, two actions that we can take are:
1. dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.2, step=0.1)
2. Increase number of end epochs
