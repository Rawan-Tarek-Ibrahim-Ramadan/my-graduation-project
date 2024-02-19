from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import normalize

app = Flask(__name__)

N_AVERAGED = 16

column_list = pd.read_csv('columnLabels.csv').columns
electrodes_list = list(pd.read_csv('columnLabels.csv').columns[4:])

# Load the pre-trained models, encoders, scalers, and other required files
model1 = pickle.load(open('best_model.pkl','rb'))
encoder = pickle.load(open('encoder.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))
top_50_feature_names = pickle.load(open('top_50_feature_names.pkl','rb'))
model2 = pickle.load(open('adboost_classifier.pkl','rb'))
model3 = load_model('gru_model.h5')
model4 = pickle.load(open("lgbb_classifier.pkl","rb"))
model5 = load_model('cnn_model.h5')
encoder1 = pickle.load(open("encoder1.pkl", "rb"))
scaler1 = pickle.load(open("scaler1.pkl", "rb"))

def predict_output1(csv_file):
    # Read the CSV file
    data = pd.read_csv(csv_file)
    
    # Preprocessing
    data['sex'] = data['sex'].map({'M': 0, 'F': 1})
    data['specific.disorder'] = encoder.transform(data[['specific.disorder']])
    df = []
    for j in list(top_50_feature_names):
        df.append(data[j])
    df = pd.concat(df, axis=1)
    df = scaler.transform(df)
    
    # Make predictions using the model and map the output
    prediction = model1.predict(df)
    case_mapping = {
        0: "Addictive disorder",
        1: "Anxiety disorder",
        2: "Healthy control",
        3: "Mood disorder",
        4: "Obsessive compulsive disorder",
        5: "Schizophrenia",
        6: "Trauma and stress related disorder",
    }
    output = case_mapping.get(prediction.tolist()[0], 'unknown')
    return output

def predict_output2(csv_file):
    # Read the CSV file
    data = pd.read_csv(csv_file)
    
    # Preprocessing
    data['log_activity'] = np.log(data['activity'] + 1)
    grouped = data.groupby('date')['log_activity'].agg(['mean', 'std', 'min', 'max', lambda x: (x == 0).sum()]).reset_index()
    grouped.columns = ['date', 'mean_log_activity', 'std_log_activity', 'min_log_activity', 'max_log_activity', 'zero_proportion_activity']
    df = grouped.drop(['date'], axis=1)
    
    # Make predictions using the model and map the output
    predictions = model2.predict(df)
    zeros_count = np.count_nonzero(predictions == 0)
    ones_count = np.count_nonzero(predictions == 1)
    if zeros_count > ones_count:
        output = "Not Depressed"
    elif ones_count > zeros_count:
        output = "Depressed"
    else:
        output = "Depressed"
        
    return output

def averaged_by_N_rows(a, n):
    shape = a.shape
    assert len(shape) == 2
    assert shape[0] % n == 0
    b = a.reshape(shape[0] // n, n, shape[1])
    mean_vec = b.mean(axis=1)
    return mean_vec

def predict_output3(csv_file):
    df = pd.read_csv(csv_file, header=None, names=column_list)

    current_sample_matrix = df[electrodes_list].values
    averaged_by_N = averaged_by_N_rows(current_sample_matrix, n=N_AVERAGED)
    averaged_by_N_big_vec = averaged_by_N.reshape(-1)
    X = averaged_by_N_big_vec.astype(np.float32)
    X_norm = normalize(X.reshape(-1, len(electrodes_list)), axis=0, norm='max')

    desired_shape = (576, len(electrodes_list))
    X_reshaped = np.zeros(desired_shape)
    if X_norm.size < np.prod(desired_shape):
        X_reshaped[:X_norm.shape[0], :X_norm.shape[1]] = X_norm
    else:
        X_reshaped = X_norm[:desired_shape[0], :desired_shape[1]]

    X_reshaped = np.expand_dims(X_reshaped, axis=0)

    predictions = model3.predict(X_reshaped)

    predictions_mapping = {
        0: "HEALTHY",
        1: "Schizophrenia"
    }
    output = 1 if predictions[0] > 0.5 else 0
    output = predictions_mapping[output]

    return output

def predict_output4(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    df = df.rename(columns={'ag+1:629e':'age'})
    df = df.rename(columns={'having.trouble.in.sleeping':'trouble.sleeping'})
    df = df.rename(columns={'having.trouble.with.work':'trouble.with.work'})
    df = df.rename(columns={'having.nightmares':'nightmares'})

    # Using model for prediction and mapping the output
    prediction = model4.predict(df)
    case_mapping = {
        0: "ADHD",
        1: "ASD",
        2: "Loneliness",
        3: "MDD",
        4: "OCD",
        5: "PDD",
        6: "PTSD",
        7: "anexiety",
        8: "bipolar",
        9: "eating disorder",
        10: "psychotic depression",
        11: "sleeping disorder" , 
    }
    output = case_mapping.get(prediction.tolist()[0], 'unknown')
    return output

def encode_categorical_features(df, columns_to_encode, encoder1):
    age = df.loc[:, "Age"].copy()

    df_categorical_encoded = encoder1.transform(df[columns_to_encode])
    dense_array = df_categorical_encoded.toarray()
    df_encoded = pd.DataFrame(dense_array, columns=encoder1.get_feature_names_out(columns_to_encode))
    df_preprocessed = pd.concat([df_encoded, age], axis=1)

    return df_preprocessed

def preprocess_data(df):
    df.drop(columns="Participant ID", inplace=True)
    categorical_columns = df.select_dtypes(include=['object']).columns
    df_preprocessed = encode_categorical_features(df, categorical_columns, encoder1)
    df_preprocessed["Age"] = scaler1.transform(df_preprocessed["Age"].values.reshape(-1, 1))
    
    return df_preprocessed

def predict_output5(csv_file):
    # Load and preprocess the data
    csv_file = request.files['file']
    data = pd.read_csv(csv_file)
    
    data_preprocessed = preprocess_data(data)
    
    # Reshape the data for CNN input
    data_reshaped = data_preprocessed.values.reshape((data_preprocessed.shape[0], data_preprocessed.shape[1], 1))
    
    # Make predictions
    predictions = model5.predict(data_reshaped)
    
    # Map predictions to labels
    label_mapping = {0: "not panic", 1: "panic"}
    output = 1 if predictions[0] > 0.5 else 0
    output = label_mapping[output]
    return output


@app.route('/')
def index():
    return render_template('home.html')
@app.route('/application1')
def page1():
    return render_template('application1.html')

@app.route('/application2')
def page2():
    return render_template('application2.html')

@app.route('/application3')
def page3():
    return render_template('application3.html')
@app.route('/application4')
def page4():
    return render_template('application4.html')
@app.route('/application5')
def page5():
    return render_template('application5.html')

@app.route('/output1', methods=['POST'])
def output1():
    if request.method == 'POST':
        file = request.files['file']
        output = predict_output1(file)
        
        if output == 'Addictive disorder':
            return render_template('addictive.html')
        elif output == 'Anxiety disorder':
            return render_template('anxiety1.html')
        elif output == 'Healthy control':
            return render_template('healthy.html')
        elif output == 'Mood disorder':
            return render_template('mood.html')
        elif output == 'Obsessive compulsive disorder':
            return render_template('OCD1.html')
        elif output == 'Schizophrenia':
            return render_template('schizophrenia.html')
        else:
            return render_template('trauma.html')
         

@app.route('/output2', methods=['POST'])
def output2():
    if request.method == 'POST':
        file = request.files['file']
        output = predict_output2(file)
        if output=='Depressed':
         return render_template('Depression.html')
        elif output=='Not Depressed':
         return render_template('healthy.html')

@app.route('/output3', methods=['POST'])
def output3():
    if request.method == 'POST':
        file = request.files['file']
        output = predict_output3(file)
        if output=='HEALTHY':
         return render_template('healthy.html')
        elif output=='Schizophrenia':
         return render_template('schizophrenia.html')
     
@app.route('/output4', methods=['POST'])
def output4():
    if request.method == 'POST':
        file = request.files['file']
        output = predict_output4(file)
        
        if output == 'ADHD':
            return render_template('ADHD.html')
        elif output == 'ASD':
            return render_template('ASD.html')
        elif output == 'Loneliness':
            return render_template('Loneliness.html')
        elif output == 'MDD':
            return render_template('MDD.html')
        elif output == 'OCD':
            return render_template('OCD1.html')
        elif output == 'PDD':
            return render_template('PDD.html')
        elif output == 'anexiety':
            return render_template('anxiety1.html')
        elif output == 'PTSD':
            return render_template('PTSD.html')
        elif output == 'bipolar':
            return render_template('bipolar.html')
        elif output == 'eating disorder':
            return render_template('eating disorder.html')
        elif output == 'psychotic depression':
            return render_template('Psychotic depression.html')
        else:
            return render_template('sleeping disorder.html')

@app.route('/output5', methods=['POST'])
def output5():
    if request.method == 'POST':
        file = request.files['file']
        output = predict_output5(file)
        if output=='not panic':
         return render_template('healthy.html')
        elif output=='panic':
         return render_template('panic.html')

if __name__ == '__main__':
    app.run()