# mnist-flask
MNIST data served/tested through a flask app. EC2 deployment. 

Uses Random Forests with 60 trees. Even though this is a simple sklearn implementation, it works better than many fancy Tensorflow implementations available on this site.  

1. Build the model by running `python build_model.py`. This will create a pickled model called `clf.pkl`. 

2. Run it as a flask app by running `python srv.py` 
