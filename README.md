# stock2
YCGN228 stock2 project from scratch using Functional Programming instead of OOP

This is a project on pulling data from Yahoo Finance, predicting the next day's price and launching it in GCP.

It is important to note that predicting the stock market is NOT the goal of this project but to simply productionize an application.

## Create Environment

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

If ou get this message: 
```
Flask has opened a server, to predict use "/get_stock_val/<ticker>"!\n
```
You succeeded!! 

Now you can edit the code and add your own improved model

# APP workflow
The app is using a flask server to process the queries. When the server receive a query on `/get_stock_val/<ticker>` where `<ticker>` is a stock name, it will:
1. Check if a model exists for this specific ticker
    - If the model exists on google storage, fetch the model.
    - If the model does not exists, download the data and train the model. Save the model on google storage.
2. Download the last days of data for this ticker (X of the model).
3. Do the inference and return it.

This is one of the most simple workflow. Feel free to change it and add complexity. However, increasing complexity comes at a cost, so, you need good reasons.

# Development workflow

You should see this process as circles. You might spend a lot of time iterating on models/strategies. However, you should always stay close to a production state where the code can run on GCP. To do so, I recommend baby steps and make sure your changes will not break the app functionality.  

# Develop and test your code
If you want to change the code and create your own version:
1. Make your changes in the code the way you want.
2. Run `python app.py` and use `curl http://localhost:8080/[name_of_your_end_point]` to test the endpoint. You can run the server from your favorite IDE. This will help to debug.

## Build and test the docker image

- Build a docker image on your local machine:  
```bash
docker build . -f Dockerfile -t my_image
```
- Run the docker image:
```bash
docker run -p 8080:8080 -v $GOOGLE_APPLICATION_CREDENTIALS:/creds.json -e GOOGLE_APPLICATION_CREDENTIALS=/creds.json my_image
```
- You can now test if the app is working using `curl`:
```bash
curl http://0.0.0.0/[name_of_your_end_point]
```
- (Optional) If for some reasons, you want to see what is going on inside the docker, you can start it in an interacting mode:
```bash
docker run -it -p 8080:8080 -v $GOOGLE_APPLICATION_CREDENTIALS:/creds.json -e GOOGLE_APPLICATION_CREDENTIALS=/creds.json my_image /bin/bash

```

## Using Evaluation.py to Evaluate Model

- This script allows testing of the model using the Balanced Accuracy Metric. It is setup to be used with 1 model per ticker on S&P 500
- If the model is changed the script will need to be updated
- Because this file is not at the main location you may run into trouble running it, in terminal run
'''
export PYTHONPATH='/<PATH TO YOUR PROJECT>/stock2'
python src/evaluations/model_evaluation.py   
'''
Then it should run against all stocks on the S&P500 using a CSV in the evaluation folder.
Theres a script in notebook format in the folder to download the S&P500 to a CSV if you need.

## Deploy your app

1. Push your code. If you are set up correctly, you should push the changes in your code and it will trigger a new build. You can check the status of your build on the Google build console.
2. The build should have created a new image. You can look at it in the 'Artifacts' tab.
3. If you click on the link at the right of the image (view), you will open a new tab (container registry)  
4. You can click on 'Deploy' and follow the instructions.

If every thing goes well, you should be redirected to a page where you can retrieve the URL of the instance.
5. On your machine, you can use curl to test your app:
```bash
curl https://[your_URL].run.app/[name_of_your_end_point]... 
```
