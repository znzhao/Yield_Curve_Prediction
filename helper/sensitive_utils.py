from github import Github
from helper.utils import findAllFile
import streamlit as st
import datetime
import re
import requests

githubToken = st.secrets["githubToken"]
fred_api_key = st.secrets["fred_api_key"]

def pushToGithub(path):
    g = Github(githubToken)
    repo = g.get_repo('znzhao/Yield_Curve_Predition')
    with open(path, 'r') as file:
        data = file.read()
        try:
            sha = repo.get_contents(path, ref='main').sha
            repo.update_file(path = path[2:], message = 'update file {}'.format(path), content = data, branch = 'main', sha = sha)
        except:
            repo.create_file(path = path[2:], message = 'update file {}'.format(path), content = data, branch = 'main')

def publishData(base):
    print('Publishing to Github...')
    for f in findAllFile(base):
        print(f)
        pushToGithub(f)
        print('Publishing file {}...'.format(f)+' '*20, end='\r')
    print('Done!' + ' '*20)

def logGithub():
    g = Github(githubToken)
    repo = g.get_repo('znzhao/Yield_Curve_Prediction')
    path = 'MktData/log.txt'
    logs = str(datetime.datetime.now().date())
    try:
        sha = repo.get_contents(path, ref='main').sha
        repo.update_file(path = path, message = 'update file {}'.format(path), content = logs, branch = 'main', sha = sha)
    except:
        repo.create_file(path = path, message = 'update file {}'.format(path), content = logs, branch = 'main')

def getLog():
    try: 
        logpath = 'https://raw.githubusercontent.com/znzhao/Yield_Curve_Prediction/main/MktData/log.txt'
        response = str(requests.get(logpath).content)
        logdate = datetime.datetime.strptime(re.findall(r'[\d-]+', response)[0], '%Y-%m-%d')
    except:
        logdate = datetime.datetime.strptime('2023-08-25', '%Y-%m-%d')
    return logdate

def pushToGithub(path, content):
    g = Github(githubToken)
    repo = g.get_repo('znzhao/Yield_Curve_Prediction')
    try:
        sha = repo.get_contents(path, ref='main').sha
        repo.update_file(path = path, message = 'update file {}'.format(path), content = content, branch = 'main', sha = sha)
    except:
        repo.create_file(path = path, message = 'update file {}'.format(path), content = content, branch = 'main')
    print('Done: {}'.format(path))

if __name__ == "__main__":
    publishData('./MktData')
