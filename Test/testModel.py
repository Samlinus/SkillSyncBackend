from joblib import load
# Create your views here.
from Modules.classfiles import Model
user_df = load('D:\\DjangoProject\\finalproject\\Modules\\userdata.pickle')
model = load('D:\\DjangoProject\\finalproject\\Modules\\model.joblib')
print(user_df)
