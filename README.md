# ALFAH -- A Toolkit for Annotating Linguistic Features in African American Oral Histories

## LREC

Publication on habitual be model work and development. Folder contains research paper and presentation.

Link to LREC code for this tagger: https://github.com/wilermine/CoLing-LREC-HabitualBe/tree/main

Sample code for reformatting data files from format used for Esemble model to format needed in my implementation of fairseq. 

Sample input and output file for Transformer.

Doing this reformatting is not absolutely necessary. You may be able to adjust fairseq commands instead. Refer to its documentation: https://fairseq.readthedocs.io/en/latest/overview.html

## habitual_be : Habitual Be Tagger

Code and documentation listed to recreate and use model. Contains minor differences from LREC code tailored to functional goals (e.g. increase in habitual recall threshold). Aditionally contains a file to utilize the model to tag new sentences.

## multiple_negation : Multiple Negation Tagger

Ruled-based tagger to identify the presence of the Multiple Negation feature at high precision and recall. 

## person_number_disagreement : Person Number Disagreement Tagger

Ruled-based tagger to identify the presence of the Person Number Disagreement feature at high recall. In-progress.

## perfect_done : Perfect Done Tagger

Ruled-based tagger to identify the presence of the Perfect Done feature. In preliminary stages; incomplete.

## remote_past_bin : Remote Past Bin Tagger

Ruled-based tagger to identify the presence of the Remote Past Bin feature. In preliminary stages; incomplete.

## null_copula : Null Copula Tagger

Ruled-based tagger to identify the presence of the Null Copula feature. In preliminary stages; incomplete.

## existensial_it_dey : Existential It/Dey Tagger

Ruled-based tagger to identify the presence of the Existential It/Dey feature. In preliminary stages; planning stage.

## annotation_guidelines : Annotation Guidelines

Guidelines and examples for annotating Habitual Be, Multiple Negation, Person Number Disagreement, and other AAE features. 
