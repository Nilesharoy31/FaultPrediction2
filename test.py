import requests

# Assuming you have the input_data list with your 21 input features
input_data = [24,6,6,2,72,346.13,0.06,17.33,19.97,5999.58,0.12,333.31,0,0,3,0,16,12,46,26,11]

url = 'http://localhost:5000/predict'
data = {'input': input_data}

response = requests.post(url, json=data)

if response.status_code == 200:
    response_json = response.json()
    print("Response:", response_json)
    prediction = response_json.get('output')
    if prediction is not None:
        print(f'Predicted class: {prediction}')
    else:
        print('Error: "output" key not found in the response.')
else:
    print('Error occurred:', response.json()['error'])

