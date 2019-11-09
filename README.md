# Credit-Card-fraud-detection-application
The intention of this project is to build a simple API that is in the form of a web application which will predict whether a particular credit card transaction is fraud or non fraud. The UI takes various parameters from the user such as time of transaction, amount and various other unnamed parameters which are not disclosed because of security reasons. There is an additional feature of this application which sends an automated mail to the concerned user if the transaction was a fraud or non fraud one. 

Prerequisites (python libraries and frameworks):

Pandas (for Machine Leraning Model), Scikit Learn, Flask (for API), Imblearn (for generating synthetic datapoints)

Project files involved:

credit_card_fraud.py : This file contains the code which fits the input data in a ML algorithm and predicts if the transaction was a fraud or not. The prediction is in the form of ‘0’ or ‘1’ depending on the outcome. The pickle function converts the code into a serialized object.

flask3.py : This file contains the code which first imports the serialized object (converted through pickle previously). The input values taken from the user by requests.form and sends the predicted outcome to the user interface using the render_template function. The code also simultaneously sends an alert mail to the customer if the transaction was a fraud or non fraud.

UI.html: This file is the frontend html page which acts as an interface to the user where the input parameters are taken and the predicted result is displayed.

The dataset can be downloaded from kaggle site
