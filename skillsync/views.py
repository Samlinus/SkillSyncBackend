from django.http import JsonResponse
from joblib import load

# Create your views here.
model = load('D:\\DjangoProject\\finalproject\\Modules\\model.joblib')


def post(request):
    if request.method == 'POST':
        print('Request detected...')
        data = request.POST.get('skills')
        # Removing the square braces..
        data = data[1:-1]
        # separating skill
        skill = [skill.strip() for skill in data.split(',')]
        print(skill)
        print(data)
        response = fetchuserdata(skill)
        print(response)
        print('sending response')
        return JsonResponse(response)
    else:
        return JsonResponse({'message': 'rejected only post message is allowed'})


def fetchuserdata(data):
    similar_users, skills = model.find_similar_users(data, 10)
    return generate_json(similar_users, skills)


def generate_json(users, skills):
    data = {}
    for user, skill_set in zip(users, skills):
        data[user] = list(skill_set)
    return data
