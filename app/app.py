############################################################################################
#                                  Author: Anass MAJJI                                     #
#                               File Name: app/app.py                                 #
#                           Creation Date: april 30, 2022                                  #
#                         Source Language: Python                                          #
#         Repository:    https://github.com/amajji/Multi-label-classification.git          #
#                              --- Code Description ---                                    #
#                       Flask API code for the model deployment                            #
############################################################################################





############################################################################################
#                                   Packages                                               #
############################################################################################
import time
import os
import os.path
import time
from os import path
from pathlib import Path
import pickle
import category_encoders
import numpy
import pandas as pd
import joblib
from flask import Flask, render_template, request, send_file
from pdf2image import convert_from_path
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestClassifier




#########################################################################################
#                                Main Code                                              #
#########################################################################################
app = Flask(__name__)
app.config['DEBUG'] = True

# Path to uploader folder
path = app.config['UPLOAD_FOLDER']


def process_date(x):
    """
    This function extracts year, month, day, hour, minute, second from a datetime feature

    :param x : datetime columns
    :return: a list of extracted year, month, day, hour, minute and second
    """
    return [
        int(x.year),
        int(x.month),
        int(x.day),
        int(x.hour),
        int(x.minute),
        int(x.second),
    ]


def add_columns(df, column_, suffix):
    """
    This function add 6 feature : year, month, day, hour, minute,
    second to the dataset and drop the datetime feature

    :param df : input datafrale.
    :param columns_ : the datetime feature.
    :param sufix : suffix we want to add to new column's names.
    :return: final dataset.
    """

    # columns of the new features we want to add
    list_columns = [
        col + suffix
        for col in ["year_", "month_", "day_", "hour_", "minute_", "second_"]
    ]

    # create a new dataframe containing extracted year, month, day, hour, minue and second from column_
    df_date = pd.DataFrame(
        process_date(x) for x in pd.to_datetime(df[column_])
    ).set_axis(list_columns, axis=1)

    # Join the dataframe created with df_continuous_features
    df = pd.concat([df_date, df], axis=1)

    # drop creation_date_answer column
    df.drop(column_, axis=1, inplace=True)

    return df


# Processing dataframe
def process_dataset(df, path):
    """
    This function applies all preporcessing steps on test dataframe :
        - Drop correlated features
        - Drop columns with missing values
        - Encode categorical features with both : Target encoder and one hot encoding

    :param  df: the test dataframe
    :param path: path where model weights are saved
    :return df after preprocessing
    """

    # Columns after dropping correlated features and ones with missing values
    columns_train = [
        "month_rqt",
        "day_rqt",
        "hour_rqt",
        "minute_rqt",
        "second_rqt",
        "year_glb",
        "month_glb",
        "day_glb",
        "hour_glb",
        "minute_glb",
        "second_glb",
        "situation",
        "number_of_fruit",
        "AP",
        "location",
        "gc_label",
        "id_group",
        "fruit_situation_id",
        "hobby",
        "green_vegetables",
    ]

    # Last columns used for models
    feature_model = [
        "month_rqt",
        "day_rqt",
        "hour_rqt",
        "minute_rqt",
        "second_rqt",
        "year_glb",
        "month_glb",
        "day_glb",
        "hour_glb",
        "minute_glb",
        "second_glb",
        "situation",
        "number_of_fruit",
        "AP_f",
        "AP_t",
        "gc_label_G",
        "gc_label_A",
        "gc_label_D",
        "gc_label_C",
        "gc_label_B",
        "gc_label_H",
        "gc_label_L",
        "gc_label_F",
        "gc_label_E",
        "gc_label_I",
        "gc_label_K",
        "gc_label_J",
        "hobby_football",
        "hobby_volleyball",
        "hobby_noball",
        "green_vegetables_f",
        "green_vegetables_t",
        "id_group",
        "fruit_situation_id",
        "location",
    ]

    # Categorical features with high and low frequencies
    features_high_frequency = ["id_group", "fruit_situation_id", "location"]
    features_low_frequency = ["AP", "gc_label", "hobby", "green_vegetables"]

    # Extract Year, month, day, hour, minute, second from creation_date_answer column
    df = add_columns(df, "creation_date_answer", "asr")

    # Extract Year, month, day, hour, minute, second from creation_date_global column
    df = add_columns(df, "creation_date_global", "glb")

    # Extract Year, month, day, hour, minute, second from creation_date_request column
    df = add_columns(df, "creation_date_request", "rqt")

    # Drop correlated columns and columns wih missing values,
    df = df[columns_train]

    # Convert categorical's feature type to string
    df[["fruit_situation_id", "location"]] = df[
        ["fruit_situation_id", "location"]
    ].astype(str)

    # load Target encoder
    encoder_te = pickle.load(open(os.path.join(path, "te_encoder.pkl"), "rb"))

    # load one hot encoder
    encoder_ohe = pickle.load(open(os.path.join(path, "ohe_encoder.pkl"), "rb"))

    # Apply Target encoder on features_high_frequency
    X_test_te = encoder_te.transform(df[features_high_frequency])

    # Drop features we just encoded
    df.drop(features_high_frequency, axis=1, inplace=True)

    # Join the encoded features with the df_categorical_features
    df = pd.concat([X_test_te, df], axis=1)

    # Apply the one hot encoder
    X_test_ohe = encoder_ohe.transform(df[features_low_frequency])

    # Join the encoded features with the df_categorical_features
    df = pd.concat([X_test_ohe, df], axis=1)

    return df[feature_model]





@app.route("/")
def page_acceuil():

    return render_template("window_principale.html")


@app.route("/uploader", methods=["POST", "GET"])
def uploader():

    if request.method == "POST":

        # Get the file
        file = request.files["file_1"]

        if file.filename != "":

	        # Save it in uploader folder
            file.save(os.path.join(path, "csv_file.csv"))

	        # Read it
            csv_file = pd.read_csv(os.path.join(path, "csv_file.csv"))


            # Get the id column before processing df_test
            df_id = csv_file["id"]

            # Processing df_test using process_dataset function
            df_test = process_dataset(csv_file, path)

            # Load the random forest model
            rdf = RandomForestClassifier()
            rdf = pickle.load(open("rdf_best.pkl", 'rb'))

            # Predicted probabilities for each sample
            predicted_proba = rdf.predict_proba(df_test)

            # Create the output dataframess
            df_result = pd.DataFrame(predicted_proba, columns=["0", "1", "2", "3"])

            # Add the id column
            df_result = pd.concat([df_id, df_result], axis=1)

            df_result.to_csv(os.path.join(path, "df_result.csv"))

    # time.sleep(4)
    return render_template("window_second.html")



@app.route("/download_result", methods=["GET"])
def download_result():

    time.sleep(5)
    return send_file(os.path.join(path, "df_result.csv"), as_attachment=True)


@app.route("/Acceuil", methods=["GET"])
def acceuil():
    return render_template("window_principale.html")


# Run the application
app.run(debug=True)
