import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", type=str, default="breast_cancer")
parser.add_argument("-max_it", type=int, default=100)
parser.add_argument("-e", type=float, default=1e-5)

# Likelihood function of logistics regression
def likelihood(data, label, w):
    likelihood = -label * (np.dot(data, w)) + np.log(1 + np.exp(np.dot(data, w)))
    return np.sum(likelihood)

# Gradient of likelihood function
def gradient(data, label, w):
    gradient = np.zeros(w.shape[0])
    for x,y in zip(data,label):
        p1 = np.exp(np.dot(w, x)) / (1 + np.exp(np.dot(w, x)))    
        gradient += x * (y - p1)  
    return -gradient

# Hessian matrix of likelihood function
def hessian(data, label, w):
    hessian = np.zeros((w.shape[0], w.shape[0]))
    for x,y in zip(data,label):
        p1 = np.exp(np.dot(w, x)) / (1 + np.exp(np.dot(w, x)))
        hessian += np.reshape(x, (x.shape[0], 1)) * x * p1 * (1-p1)
    return hessian

# Solve logistics regression with newton's method
def logistic_regression(data, label, e, max_it=100):    
    # Expand data matrix
    data = np.c_[data, np.ones(data.shape[0])] 
    # Initialize coeffiencts of model y = 1 / (1 + e^(-wx))
    coef = np.zeros(data.shape[1]) 
    # Initialize step norm
    d_norm = np.inf 
    # Number of iteration
    it_count = 0 

    while d_norm > e and it_count < max_it:
        print("Step:", it_count, "Likelihood:", likelihood(data, label, coef))
        d = np.dot(np.linalg.inv(hessian(data, label, coef)), gradient(data, label, coef))
        # Update coefficient
        coef = coef - d 
        # Calculate step norm
        d_norm = np.linalg.norm(d) 
        it_count += 1

    print("Result:\n", coef)        
    return coef

# Predict novel data with trained coefficient
def predict(data, w):
    data = np.c_[data, np.ones(data.shape[0])]
    p0 = 1 / (1 + np.exp(np.dot(data, w)))
    p1 = 1 - p0
    res = np.c_[p0,p1]
    res = np.argmax(res, axis=1)
    return res

# Calculate the percision of predicted result
def score(predict, ground_truth):
    count = (predict == ground_truth).astype(int).sum()
    return (count/len(predict))

def main():
    args = parser.parse_args()

    # Load data
    print("Loading dataset: ", args.dataset)
    if args.dataset == "breast_cancer":
        data = np.genfromtxt("data/breast-cancer-wisconsin.data", delimiter=",")
        # Remove nan data samples and ID col
        data = data[~np.isnan(data).any(axis=1)][:,1:]
        np.random.shuffle(data)
        x = data[:,:-1]
        # Map label to 0,1
        label = data[:, -1]
        unique, y = np.unique(label, return_inverse=True)
    elif args.dataset == "abalone":
        data = np.genfromtxt("data/abalone.data", delimiter=",", dtype=str)
        np.random.shuffle(data)
        x = data[:,1:-1].astype(float)
        # Map ages to class 0,1
        label = data[:, -1].astype(float)
        y = (label > 10).astype(int)
    else:
        raise Exception("Dataset can only be breast_cancer or abalone")

    # Train test split
    partition = 0.8
    train_size = int(data.shape[0] * 0.8)

    train_data = x[:train_size]
    train_label = y[:train_size]

    test_data = x[train_size:]
    test_label = y[train_size:]

    # Solve model
    res = logistic_regression(train_data, train_label, e=args.e, max_it=args.max_it)

    # Predict novel data
    pred = predict(test_data, res)
    # Score
    print("Percison: ", score(pred, test_label))

if __name__ == "__main__":
    main()