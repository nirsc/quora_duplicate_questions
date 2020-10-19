import json
import os
optimizers = ['adam']
num_epochs = [1000]
model = ['siamese']
data_file = 'C:\\Users\\NIR\\PycharmProjects\\quora_duplicate_questions\\data\\quora_duplicate_questions.h5'
i=1
if not os.path.exists('..\\configs\\'):
    os.mkdir('..\\configs')


for opt in optimizers:
    for ne in num_epochs:
        for md in model:
                    d = {}
                    d['data_file'] = data_file
                    d['max_len'] = 40
                    d['model_params'] = {}
                    d['model_params']['optimizer'] = opt
                    d['model_params']['num_epochs'] = ne
                    d['model_params']['model_type'] = md

                    dict_file_name = "..\\configs\\config{}.json".format(i)
                    i += 1
                    with open(dict_file_name,'w') as f:
                        json.dump(d,f,indent = '\n')




