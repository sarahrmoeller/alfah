# **Habitual Be** 

## Habitual Rule Tagger.ipynb

Tags each sentence with the rules, e.g. '1' if present and '0' if absent. Note: Habituality is not tagged in this file.

## Habitual Rule Generator.ipynb

Generates three joblib files based on the output of Habitual Rule Tagger.ipynb: cv, habituality_model, and n_gram. 

## Use This to Predict Habituality of New Files.py

Uses the joblib files generated by Habitual Rule Generator.ipynb to make predictions about sentence habituality in a new dataset.

## Habitual Be Full and Final Documentation.docx

Provides key information about the habitual be pipeline and how to test the process.

## cv.joblib

Count vectorizer used for n_gram predictions. 

## habituality_model.joblib

Machine learning habitual be  model trained on dataset and exported for usage.

## n_gram.joblib

Machine learning habitual be model trained on n_gram data sourced from dataset and exported for usage.

## test gold standard lines+labels.csv

Dataset with gold standard (true) values of habituality for each sentence. Input into Habitual Rule Tagger.ipynb.

## test new predicted coraal_analysis_spreadsheet.csv

Dataset containing habituality predictions of sentences from test gold standard lines+labels.csv. During set-up, confirm this matches coraal_analysis_spreadsheet.csv, which outputs from Habitual Rule Generator.ipynb.

## Merge_Speaker_and_File.py

Code to match speakers with their appropriate sentences and generate new file with separate speaker column.

## new_texts_for_tagging

Folder containing data to test functionality of Use This to Predict Habituality of New Files.py to predict habituality of new datasets.

# speaker

Folder containing the speakers associated with the new_texts_for_tagging files.

## new_texts_for_tagging_speakers

Folder containing new_texts_for_tagging sentences with their appropriate speakers. Should be output from Merge_Speaker_and_File.py. 
