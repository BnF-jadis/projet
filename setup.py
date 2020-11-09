import os

os.makedirs('export', exist_ok=True)
os.makedirs('workshop', exist_ok=True)
os.makedirs('save', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('model', exist_ok=True)
os.makedirs('settings', exist_ok=True)
os.system('pip install --upgrade pip')
os.system('conda update -n base -c defaults conda')
os.system('pip install numpy')
os.system('pip install -r requirements.txt')
