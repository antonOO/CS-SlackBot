
from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer
import json

#TRAIN WITH THE DEMO RASA FILE USING SPACY
'''
training_data = load_data('restaurant/demo-rasa.json')
trainer = Trainer(RasaNLUConfig("restaurant/config_spacy.json"))
trainer.train(training_data)
trainer.persist('restaurant/') '''
model_directory = "restaurant/default/model_20171115-192654"

#LOAD THE MODEL USING THE METADATA INTERPETER
from rasa_nlu.model import Metadata, Interpreter
interpreter = Interpreter.load(model_directory, RasaNLUConfig("restaurant/config_spacy.json"))
interpretation = interpreter.parse("OPI")
print(interpretation['intent']['name'])
print(interpretation['intent']['confidence'])
