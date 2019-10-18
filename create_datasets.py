import json


# load ontology
ontology = json.load(open('json/ontology.json', 'rb'))
ontology_by_id = {obj['id']: obj for obj in ontology}

# load ground truth: [[<fs_id>, <aso_id>, <0.5/1>, <duration>], ...]
ground_truth = json.load(open('json/ground_truth_annotations_28_05_19.json', 'rb'))

def create_dataset(node_id):
    child_ids = set(ontology_by_id[node_id]['child_ids'])
    dataset = {
        'name': ontology_by_id[node_id]['name'],
        'id': node_id,
        'sound_ids': [],
        'dataset': {n_id: [] for n_id in child_ids}
    }
    for gt in ground_truth:
        if gt[1] in child_ids:
            if gt[3] <= 10:  # filter sound with duration > 10 sec
                # take only Present Predominant sounds except for Music Mood and Wind and more??
                if gt[2] == 1.0 or node_id in ('/t/dd00030', '/m/03m9d0z'):  
                    dataset['dataset'][gt[1]].append(gt[0])

    # exclude the categories which have less than 10 sounds
    for n_id in list(dataset['dataset']):
        if len(dataset['dataset'][n_id]) < 10:
            del dataset['dataset'][n_id]

    # add all ids info
    for n_id, obj in dataset['dataset'].items():
        dataset['sound_ids'] += obj
    dataset['sound_ids'] = list(set(dataset['sound_ids']))

    # TODO: exclude multi-labeled sounds?
    
    return dataset


if __name__ == '__main__':
    """
    Create the dataset json files for the selected categories
    """
#   Domestic animals, pets
#   Livestock, farm animals, working animals
#   Wild animals
#   Dog
#   Cat
#   Fowl
#   Cattle, bovinae
#   Human sounds
#   Human locomotion
#   Respiratory sounds
#   Digestive
#   Human voice
#   Speech
#   Singing
#   Human group actions
#   Hands
#   Music mood [REMOVE]
#   Musical concepts
#   Musical instrument
#   Wind instrument, woodwind instrument
#   Keyboard (musical)
#   Bowed string instrument
#   Brass instrument
#   Bell
#   Plucked string instrument
#   Guitar
#   Percussion
#   Mallet percussion
#   Domestic sounds, home sounds
#   Door
#   Tools
#   Liquid
#   Alarm
#   Explosion
#   Vehicle
#   Motor vehicle (road)
#   Car
#   Aircraft
#   Non-motorized land vehicle
#   Rail transport
#   Glass
#   Mechanisms
#   Engine
#   Natural sounds
#   Water
#   Wind

    datasets_to_create = [
        '/m/068hy',
        '/m/0ch8v',
        '/m/01280g',
        '/m/0bt9lr',
        '/m/01yrx',
        '/m/025rv6n',
        '/m/01xq0k1',
        '/m/0dgw9r',
        '/m/0bpl036',
        '/m/09hlz4',
        '/m/0160x5',
        '/m/09l8g',
        '/m/09x0r',
        '/m/015lz1',
        '/t/dd00012',
        '/m/0k65p',
#        '/t/dd00030',
        '/t/dd00027',
        '/m/04szw',
        '/m/085jw',
        '/m/05148p4',
        '/m/0l14_3',
        '/m/01kcd',
        '/m/0395lw',
        '/m/0fx80y',
        '/m/0342h',
        '/m/0l14md',
        '/m/0j45pbj',
        '/t/dd00071',
        '/m/02dgv',
        '/m/07k1x',
        '/m/04k94',
        '/m/07pp_mv',
        '/m/014zdl',
        '/m/07yv9',
        '/m/012f08',
        '/m/0k4j',
        '/m/0k5j',
        '/t/dd00061',
        '/m/06d_3',
        '/m/039jq',
        '/t/dd00077',
        '/m/02mk9',
        '/m/059j3w',
        '/m/0838f',
        '/m/03m9d0z',                
    ]
    
    all_sound_ids = []
    
    for idx, dataset_id in enumerate(datasets_to_create):
        dataset = create_dataset(dataset_id)
        for _, sound_ids in dataset['dataset'].items():
            all_sound_ids += sound_ids
        json.dump(dataset, open('datasets/{0}.json'.format(idx), 'w'))
    json.dump(list(set(all_sound_ids)), open('all_sound_ids.json', 'w'))
        

        
"""
DOWNLOAD FREESOUND SOUNDS:
freesound-python
import manager
c = manager.Client()
b = c.new_basket()
b.push_list_id(all_sound_ids)
Bar = manager.ProgressBar(len(b), 30, 'Downloading...')
for idx, s in enumerate(b.sounds):
    try:
        s.retrieve('sound_files_clustering', str(s.id) +'.' + s.type)
    except:
        pass
    Bar.update(idx+1)
"""

"""
How to get Freesound analysis stats:
freesound-python
import manager
c = manager.Client()
b = c.new_basket()
b.push_list_id(all_sound_ids)
b.add_analysis_stats()
for idx, a in enumerate(b.analysis_stats):
    try:
        json.dump(a.as_dict(), open(str(b.ids[idx]) + '.json', 'w'))
    except:
        pass

"""
