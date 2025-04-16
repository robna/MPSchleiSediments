# Note: This file is just needed for streamlit app to run when deployed Streamlit app platform.
#       Not needed for local run.
#       Adapted from: https://github.com/IndigoWizard/Shinosuke-lab/tree/main

import os
import sys

# get the path of your current script (main.py in here)
script_path = os.path.dirname(os.path.abspath(__file__))

# add 'analysis' folder to sys.path
app_folder = os.path.join(script_path, 'analysis')
sys.path.append(app_folder)

# import and run the Streamlit app from app.py that lives in analysis folder
import app

if __name__ == '__main__':
    os.chdir(app_folder)  # change the working directory to the app folder
    app.main()  # run the Streamlit app
