# This program is programmed by Wonjun Yi (KAIST,20208220) for team project.

import os
import json
import pandas
import numpy
import shutil

# move to directory

os.chdir('./maestro-v3.0.0-midi')
os.chdir('maestro-v3.0.0')

# read json file

json_file = open('maestro-v3.0.0.json')
json_dict = json.load(json_file)
json_file.close()

# analyze json file in pandas dataframe

json_panda = pandas.DataFrame.from_dict(json_dict)
composer_info = json_panda['canonical_composer'].value_counts()
composers = composer_info.keys()
n_composer = len(composer_info)

composer_info.to_csv('composer_info.csv')

composer_plot = composer_info.plot(kind = 'bar', title = 'composer info', xlabel = 'composer', ylabel = 'file #', figsize = (10,10))
composer_figure = composer_plot.get_figure()
composer_figure.tight_layout()
composer_figure.savefig('composer_info.jpg', dpi = 1000)
composer_figure.savefig('composer_info.svg', dpi = 1000)
composer_figure.clf()

# analyze composers_by_era.xlsx
era2composer = pandas.read_excel('./composers_by_era.xlsx')
music_era = era2composer.keys()
n_eras = []
for i_era in range(len(music_era)):
    composer_in_i_era = era2composer[music_era[i_era]].dropna()
    n_era = 0
    for icomposer in composer_in_i_era:
        n_era += composer_info[icomposer]
    n_eras.append(n_era)
music_era_tot = pandas.Series(n_eras,index = music_era)
music_era_tot.name = 'music_era'
music_era_tot.to_csv('era_num.csv')

era_plot = music_era_tot.plot(kind = 'bar', title = 'era info', xlabel = 'era', ylabel = 'file #', figsize = (10,10))
era_figure = era_plot.get_figure()
era_figure.tight_layout(pad = 3)
era_figure.savefig('era_info.jpg', dpi = 1000)
era_figure.savefig('era_info.svg', dpi = 1000)

#for top_composer in composers[0:top]:
for top_composer in composers:
    top_composer_ = '_'.join(top_composer.split(' '))
    if '/' in top_composer_:
        top_composer_ = 'and'.join(top_composer_.split('/'))
    print(top_composer)
    os.makedirs(top_composer_, exist_ok=True)
    os.chdir(top_composer_)
    titles = json_panda[json_panda['canonical_composer'] == top_composer]['canonical_title']
    titles.to_csv('titles.csv')
    files = json_panda[json_panda['canonical_composer'] == top_composer]['midi_filename']
    files.to_csv('files.csv')
    for ifile in files:

        shutil.copy2('../'+ifile, ifile.split('/')[-1])

    # Also have to save new .json file for train.py
    re_json_ = json_panda[json_panda['canonical_composer'] == top_composer]
    re_json_.index = list(range(len(files)))

    n_train = int(len(files) * 0.8)
    n_valid = int(len(files) * 0.1)
    re_json_['split'].loc[0:n_train] = 'train'
    re_json_['split'].loc[n_train:n_train + n_valid] = 'validation'
    re_json_['split'].loc[n_train+n_valid:len(files)] = 'test'
    re_json = json.dumps(re_json_.to_dict())
    with open("info.json","w") as outfile:
        outfile.write(re_json)

    os.chdir('..')


for i_era in music_era:
    print(i_era)
    sw = 0
    titles = 0
    files = 0
    re_json_ = 0
    composer_in_i_era = era2composer[i_era].dropna()
    for top_composer in composer_in_i_era:
        top_composer_ = '_'.join(top_composer.split(' '))
        if '/' in top_composer_:
            top_composer_ = 'and'.join(top_composer_.split('/'))


        if sw == 0:
            titles = json_panda[json_panda['canonical_composer'] == top_composer]['canonical_title']
            files = json_panda[json_panda['canonical_composer'] == top_composer]['midi_filename']
            re_json_ = json_panda[json_panda['canonical_composer'] == top_composer]
            re_json_.index = list(range(len(files)))
            n_train = int(len(files) * 0.8)
            n_valid = int(len(files) * 0.1)
            re_json_['split'].loc[0:n_train] = 'train'
            re_json_['split'].loc[n_train:n_train + n_valid] = 'validation'
            re_json_['split'].loc[n_train + n_valid:len(files)] = 'test'
            tot_json = re_json_
            sw = 1
        else:
            loc_titles = json_panda[json_panda['canonical_composer'] == top_composer]['canonical_title']
            loc_files = json_panda[json_panda['canonical_composer'] == top_composer]['midi_filename']
            titles = pandas.concat([titles,loc_titles])
            files = pandas.concat([files, loc_files])
            re_json_ = json_panda[json_panda['canonical_composer'] == top_composer]
            re_json_.index = list(range(len(loc_files)))
            n_train = int(len(files) * 0.8)
            n_valid = int(len(files) * 0.1)
            re_json_['split'].loc[0:n_train] = 'train'
            re_json_['split'].loc[n_train:n_train + n_valid] = 'validation'
            re_json_['split'].loc[n_train + n_valid:len(files)] = 'test'
            tot_json = pandas.concat([tot_json,re_json_])


    re_json = json.dumps(tot_json.to_dict())
    os.makedirs(i_era, exist_ok=True)
    os.chdir(i_era)
    titles.to_csv('titles.csv')
    files.to_csv('files.csv')
    for ifile in files:

        shutil.copy2('../'+ifile, ifile.split('/')[-1])
    with open("info.json","w") as outfile:
        outfile.write(re_json)

    os.chdir('..')



