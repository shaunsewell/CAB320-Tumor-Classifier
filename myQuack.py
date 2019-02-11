#imports
import numpy as np
from sklearn import preprocessing,neighbors,svm,tree,naive_bayes,metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score,cross_val_predict
import scipy

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Prediction_table = []

def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ (01234567, 'Shaun', 'Sewell') ]
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''
    try:
        fCols = list(range(2,32))
        #ID = genfromtxt(dataset_path,delimiter=',',dtype=int,usecols=0)
        #fValues = genfromtxt(dataset_path,delimiter=',',dtype=float,usecols=fCols)
        X = np.loadtxt(dataset_path,delimiter=',',dtype=float,usecols=fCols)
        y = np.loadtxt(dataset_path,delimiter=',',dtype=bytes,usecols=1).astype(str)
        return X,y
    except:
        raise

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NB_classifier(X_training, y_training):
    '''  
    Build a Naive Bayes classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    #init 3 different NB clf and store in a np array
    clf_array = np.array([naive_bayes.GaussianNB(),naive_bayes.MultinomialNB(),naive_bayes.BernoulliNB()])
    CV_scores = np.empty(3)
    #loop thru each clf in order
    for i in range(0,3):
        #CV each clf
        clf_CV = cross_val_score(clf_array[i],X=X_training,y=y_training, cv=4)
        #store average of the training
        CV_scores[i] = clf_CV.mean()
    #pic the clf with the highest score
    index = np.argmax(CV_scores)
    #train the clf on the whole training set
    clf = clf_array[index].fit(X_training,y_training)
    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DT_classifier(X_training, y_training):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    # Parameters list for tuning hyper-parameters of the classifier
    parameters = {'max_depth':list(range(1,100,1))} #values to try for n_neighbors
    # Init a KNN clf
    DT_clf = tree.DecisionTreeClassifier()
    #Tuning the Hyper-parameters
    clf = GridSearchCV(DT_clf,parameters,scoring='f1_macro',cv=4)   #Build all clf with different parameters
    clf.fit(X_training,y_training)        #train the clf's 
    #build a new clf using the best params found
    best_clf = tree.DecisionTreeClassifier(max_depth=clf.best_params_['max_depth'])
    best_clf.fit(X_training,y_training)    #train the model using the whole training set
    return best_clf       #return return teh finalised clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NN_classifier(X_training, y_training):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    
    # Parameters list for tuning hyper-parameters of the classifier
    parameters = {'n_neighbors':list(range(1,100,1))} #values to try for n_neighbors
    # Init a KNN clf
    NN_clf = neighbors.KNeighborsClassifier()
    #Tuning the Hyper-parameters
    clf = GridSearchCV(NN_clf,parameters,scoring='f1_macro',cv=4)   #Build all clf with different parameters
    clf.fit(X_training,y_training)        #train the clf's 
    #build a new clf using the best params found
    best_clf = neighbors.KNeighborsClassifier(n_neighbors=clf.best_params_['n_neighbors'])
    best_clf.fit(X_training,y_training)    #train the model using the whole training set
    return best_clf       #return return teh finalised clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SVM_classifier(X_training, y_training):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    
    # Parameters list for tuning hyper-parameters of the classifier
    parameters = {'C':list(range(1,100,1))} #values to try for C
    # Init an svm
    svm_clf = svm.SVC(kernel='linear')
    #Tuning the Hyper-parameters
    clf = GridSearchCV(svm_clf,parameters,scoring='f1_macro',cv=4)   #Build all clf with different parameters
    clf.fit(X_training,y_training)        #train the clf's 
    #build a new clf using the best params found
    best_clf = svm.SVC(C=clf.best_params_['C'])
    best_clf.fit(X_training,y_training)    #train the model using the whole training set
    return best_clf       #return return teh finalised clf
   
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_table(SVM,NN,DT,NB,X_data,y_data):
    '''  
    Build a table of accuracies of the supplied trained classifiers.

    @param 
	X_data: X_data[i,:] is the ith example
	y_data: y_data[i] is the class label of X_data[i,:]
    SVM: Trained Support Vector Machine classifier
    NN: Trained Nearest Neighbor classifier
    DT: Trained Decision Tree classifier
    NB: Trained Naive-Bayes classifier
    @return
	table : The table of accuracies built in this function.
    '''

    #build table
    top_row = "Classifier\t Accuracy\n"
    #predict using svm
    y_data_pred = SVM.predict(X_data)
    SVM_row = "\n   SVM\t\t    %0.2f\n" % accuracy_score(y_data, y_data_pred)
    #predict using NN
    y_data_pred = NN.predict(X_data)
    NN_row = "    NN\t\t    %0.2f\n" % accuracy_score(y_data, y_data_pred)
    #predict using DT
    y_data_pred = DT.predict(X_data)
    DT_row = "    DT\t\t    %0.2f\n" % accuracy_score(y_data, y_data_pred)
    #predict using NB
    y_data_pred = NB.predict(X_data)
    NB_row = "    NB\t\t    %0.2f\n" % accuracy_score(y_data, y_data_pred)
    table = top_row + SVM_row + NN_row + DT_row + NB_row
    return table
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
def recommend_classifier(SVM,NN,DT,NB,X_testing,y_testing):
    '''  
    Identifies the highest scoring classifier for the testing data.

    @param 
	X_data: X_data[i,:] is the ith example
	y_data: y_data[i] is the class label of X_data[i,:] 
    SVM: Trained Support Vector Machine classifier 
    NN: Trained Nearest Neighbor classifier 
    DT: Trained Decision Tree classifier 
    NB: Trained Naive-Bayes classifier 
    @return
	A string representing the type of classifier recommended
    '''
    #construct predictions
    SVM_pred = SVM.predict(X_testing)
    NN_pred = NN.predict(X_testing)
    DT_pred = DT.predict(X_testing)
    NB_pred = NB.predict(X_testing)
    
    scores = np.array([accuracy_score(y_testing,SVM_pred),accuracy_score(y_testing,NN_pred),accuracy_score(y_testing,DT_pred),accuracy_score(y_testing,NB_pred)])
    index = np.argmax(scores)
    if index == 0:    #SVM
        return "Support Vector Machine"
    elif index == 1:    #Nearest Neighbor
        return "Nearest Neighbor"
    elif index == 2:    #Decision Tree
        return "Decision Tree"
    else:
        return "Naive-Bayes"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    #Loading dataset
    X,y = prepare_dataset("medical_records.data")
    #Preprocess X so that the data is between 0-1 and normalised
    X_scaled = preprocessing.scale(X)
    X_scaled = preprocessing.minmax_scale(X_scaled)
        
    #split into train and test sets, training set will be further split when building each classifier
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=(0.15), random_state=22)
    
    #Build classifiers
    SVM = build_SVM_classifier(X_train, y_train)
    NN = build_NN_classifier(X_train, y_train)
    DT = build_DT_classifier(X_train, y_train)
    NB = build_NB_classifier(X_train, y_train)
    
    #display results
    print("Prediction Accuracy - Training set")
    print(build_table(SVM, NN, DT, NB, X_train, y_train))
    print("Prediction Accuracy - Testing set")
    print(build_table(SVM, NN, DT, NB, X_test, y_test)) 
    #Recommend a Classifier
    print("Based on the data set provided the %s classifier is recommended." % recommend_classifier(SVM,NN,DT,NB,X_test,y_test))
