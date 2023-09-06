from github import Github
from helper.utils import findAllFile
def pushToGithub(path):
    githubToken = "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"
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

if __name__ == "__main__":
    publishData('./MktData')