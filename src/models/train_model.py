from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle


# Function to train the model
def train_Neuralmodel(df):

    x = df.drop(['Admit_Chance'], axis=1)
    y = df['Admit_Chance']
    # Splitting the data into training and testing sets
    xtrain, xtest, ytrain, ytest =  train_test_split(x, y, test_size=0.2, random_state=123, stratify=y)

    # fit calculates the mean and standard deviation
    scaler = MinMaxScaler()
    scaler.fit(xtrain)
    xtrain_scaled = scaler.transform(xtrain)
    xtest_scaled = scaler.transform(xtest)


    Neuralmodel = MLPClassifier(hidden_layer_sizes=(3,3), batch_size=50, max_iter=200, random_state=123)
    Neuralmodel.fit(xtrain_scaled,ytrain)
    
    # Save the trained model
    with open('models/Neuralmodel.pkl', 'wb') as f:
        pickle.dump(Neuralmodel, f)

    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return Neuralmodel, xtest_scaled, ytest
