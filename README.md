# stock2
YCGN228 stock2 project from scratch using Functional Programming instead of OOP

This is a project on pulling data from Yahoo Finance, predicting the next day's price and launching it in GCP.

It is important to note that predicting the stock market is NOT the goal of this project but to simply productionize an application.

## Step 1: Create Environment

- Ensure you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) & [Docker](https://docs.docker.com/get-docker/) installed
- Setup a GCP account and project
- Hook your GitHub repo to your GCP account [link](https://cloud.google.com/build/docs/automating-builds/create-manage-triggers)
- Download and create GOOGLE_APPLICATION_CREDENTIALS env variable [Get the credentials (json)](https://developers.google.com/workspace/guides/create-credentials#create_credentials_for_a_service_account)

Local Machine Setup:
1. Fork this repository in your github account
2. [Create your environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) using `scripts/environment.yml`.  
3. Activate the environment. `conda activate stock2`
4. Run the app: `python app.py` 
5. Test the endpoint in a separated shell `curl http://localhost:8080/`

