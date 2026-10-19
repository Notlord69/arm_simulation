import os
import neuprint as neu

c = neu.Client('https://neuprint.janelia.org',
               dataset='male-cns:v1.0',
               token=os.environ['NEUPRINT_TOKEN'])

print("Connected to:", c.dataset)

neurons, roi = neu.fetch_neurons(neu.NeuronCriteria(type='DNa01'))
print(f"Found {len(neurons)} DNa01 neurons")
print(neurons[['bodyId', 'type', 'instance']].head())
