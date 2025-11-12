import json

# Load model.json
with open('models/baseline_children_asd_tfjs/model.json', 'r') as f:
    data = json.load(f)

print('Layer names:')
for layer in data['modelTopology']['config']['layers']:
    if layer['class_name'] != 'InputLayer':
        print(f'  - {layer["config"]["name"]} ({layer["class_name"]})')

print('\nWeight names (first 15):')
for w in data['weightsManifest'][0]['weights'][:15]:
    print(f'  - {w["name"]}')
